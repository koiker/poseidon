import os
import sys
import torch
import torch.nn as nn
import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict
from .backbones import Backbones
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
import cv2
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch.nn.functional as F
from mmpose.apis import init_model
import math


from posetimation import get_cfg, update_config 

class AdaptiveFrameWeighting(nn.Module):
    def __init__(self, embed_dim, num_frames):
        super(AdaptiveFrameWeighting, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        self.frame_quality_estimator = nn.Sequential(
            nn.Conv2d(self.embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
                
    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, height, width]
        batch_size, num_frames, embed_dim, height, width = x.shape
        
        # Estimate quality for each frame
        x_reshaped = x.view(batch_size * num_frames, embed_dim, height, width)
        quality_scores = self.frame_quality_estimator(x_reshaped).view(batch_size, num_frames)
        
        # Normalize scores
        weights = F.softmax(quality_scores, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Weight frames
        weighted_x = x * weights
        
        return weighted_x, weights.squeeze()

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_frames):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.num_frames = num_frames

        # Central frame bias
        self.central_frame_bias = nn.Parameter(torch.zeros(num_frames))

        # Initialize central frame bias to emphasize the central frame
        center = num_frames // 2
        self.central_frame_bias.data[center] = 1.0

    def forward(self, src):
        # src shape: [num_frames, batch_size * H * W, embed_dim]

        # Apply central frame bias to attention weights
        num_frames, N, embed_dim = src.shape
        central_bias = self.central_frame_bias.unsqueeze(1).unsqueeze(2)  # [num_frames, 1, 1]
        src_with_bias = src + central_bias

        attn_output, _ = self.self_attn(src_with_bias, src_with_bias, src_with_bias)

        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src2 = self.dropout2(src2)
        src = self.norm2(src + src2)
        return src

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_frames):
        super(TemporalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, num_frames)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, H, W]
        batch_size, num_frames, embed_dim, H, W = x.shape
        x = x.view(batch_size, num_frames, embed_dim, -1)  # Flatten spatial dimensions
        x = x.permute(1, 3, 0, 2)  # [num_frames, H*W, batch_size, embed_dim]
        x = x.reshape(num_frames, batch_size * H * W, embed_dim)  # Merge batch and spatial dimensions

        for layer in self.layers:
            x = layer(x)

        # Reshape back to original dimensions
        x = x.view(num_frames, batch_size, H * W, embed_dim)
        x = x.permute(1, 0, 3, 2)  # [batch_size, num_frames, embed_dim, H*W]
        x = x.view(batch_size, num_frames, embed_dim, H, W)

        return x

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=256, num_layers=1):
        super(Poseidon, self).__init__()

        self.device = device
        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        
        self.backbone = self.model.backbone

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE

        # Model parameters
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.is_train = phase == 'train'

        # Adaptive Frame Weighting
        self.adaptive_weighting = AdaptiveFrameWeighting(self.embed_dim, self.num_frames)

        # Temporal Transformer with Central Frame Attention
        self.temporal_transformer = TemporalTransformer(self.embed_dim, self.num_heads, num_layers, self.num_frames)

        # Layer normalization
        self.layer_norm = nn.LayerNorm([self.embed_dim, 16, 12])

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n")
        print(f"Poseidon total parameters: {round(self.count_parameters() / 1e6, 1)} M\n")

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone feature extraction
        model_output = self.backbone(x)[0]

        # Reshape model output
        H_feat, W_feat = model_output.shape[2], model_output.shape[3]
        x = model_output.view(batch_size, num_frames, self.embed_dim, H_feat, W_feat)

        # Compute temporal differences
        motion_features = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]
        motion_features = torch.cat([motion_features, torch.zeros_like(motion_features[:, :1])], dim=1)
        x = x + motion_features

        # Adaptive Frame Weighting
        x, frame_weights = self.adaptive_weighting(x)

        # Temporal Transformer
        x = self.temporal_transformer(x)

        # Layer Norm on central frame
        x = self.layer_norm(x[:, num_frames // 2])

        # Deconvolution layers
        x = self.deconv_layer(x)

        # Final layer
        x = self.final_layer(x)

        return x

    def set_phase(self, phase):
        self.phase = phase
        self.is_train = True if phase == TRAIN_PHASE else False

    def get_phase(self):
        return self.phase

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        N = output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            norm_factor=np.ones((N, 2), dtype=np.float32))

        return avg_acc



