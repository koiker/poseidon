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
            nn.Conv2d(self.embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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


class TemporalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TemporalCrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, H, W]
        batch_size, num_frames, embed_dim, H, W = x.shape
        x = x.view(batch_size, num_frames, embed_dim, -1)  # Flatten spatial dimensions
        x = x.permute(0, 3, 2, 1)  # [batch_size, H*W, embed_dim, num_frames]
        x = x.reshape(batch_size * H * W, embed_dim, num_frames)  # Merge batch and spatial dimensions
        x = x.permute(2, 0, 1)  # [num_frames, batch_size * H * W, embed_dim]

        # Apply cross-attention between frames
        x, _ = self.cross_attention(x, x, x)  # [num_frames, batch_size * H * W, embed_dim]

        x = x.permute(1, 2, 0)  # [batch_size * H * W, embed_dim, num_frames]
        x = x.view(batch_size, H * W, embed_dim, num_frames)
        x = x.permute(0, 3, 2, 1)  # [batch_size, num_frames, embed_dim, H*W]
        x = x.view(batch_size, num_frames, embed_dim, H, W)

        return x

class HierarchicalTemporalEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(HierarchicalTemporalEncoder, self).__init__()

        # Hierarchical Temporal Transformer layers
        self.short_term_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=num_layers
        )
        self.long_term_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=num_layers
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, H, W]
        batch_size, num_frames, embed_dim, H, W = x.shape
        x = x.view(batch_size, num_frames, embed_dim, -1)  # Flatten spatial dimensions
        x = x.permute(0, 3, 2, 1)  # [batch_size, H*W, embed_dim, num_frames]
        x = x.reshape(batch_size * H * W, embed_dim, num_frames)  # Merge batch and spatial dimensions
        x = x.permute(2, 0, 1)  # [num_frames, batch_size * H * W, embed_dim]

        # Short-term encoding: Apply transformer over small temporal windows
        window_size = 3  # E.g., process 3-frame windows
        x_short_term = []
        for i in range(0, num_frames, window_size):
            short_term_out = self.short_term_transformer(x[i:i + window_size])
            x_short_term.append(short_term_out)
        
        x_short_term = torch.cat(x_short_term, dim=0)  # [num_frames, batch_size * H * W, embed_dim]

        # Long-term encoding: Process the short-term encoded representations
        x_long_term = self.long_term_transformer(x_short_term)

        x = x_long_term.permute(1, 2, 0)  # [batch_size * H * W, embed_dim, num_frames]
        x = x.view(batch_size, H * W, embed_dim, num_frames)
        x = x.permute(0, 3, 2, 1)  # [batch_size, num_frames, embed_dim, H*W]
        x = x.view(batch_size, num_frames, embed_dim, H, W)

        return x


class TemporalTransformerWithCrossAttentionAndHierarchical(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TemporalTransformerWithCrossAttentionAndHierarchical, self).__init__()
        
        # Temporal Cross-Attention
        self.temporal_cross_attention = TemporalCrossAttention(embed_dim, num_heads)
        
        # Hierarchical Temporal Encoding
        self.hierarchical_temporal_encoder = HierarchicalTemporalEncoder(embed_dim, num_heads, num_layers)
        
    def forward(self, x):
        # Apply temporal cross-attention
        x = self.temporal_cross_attention(x)
        
        # Apply hierarchical temporal encoding
        x = self.hierarchical_temporal_encoder(x)
        
        return x

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=384, num_layers=1):
        super(Poseidon, self).__init__()

        self.device = device
        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        
        self.backbone = self.model.backbone


        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE

        # Model parameters
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_frames = cfg.WINDOWS_SIZE
        self.is_train = True if phase == 'train' else False

        # Adaptive Frame Weighting
        self.adaptive_weighting = AdaptiveFrameWeighting(self.embed_dim, self.num_frames)

        # Temporal Transformer with Cross-Attention and Hierarchical Encoding
        self.temporal_transformer = TemporalTransformerWithCrossAttentionAndHierarchical(self.embed_dim, self.num_heads, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm([self.embed_dim, 24, 18])

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")
        print(f"Poseidon total parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

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
        x = model_output.view(batch_size, num_frames, self.embed_dim, 24, 18)

        # Adaptive Frame Weighting
        x, frame_weights = self.adaptive_weighting(x)

       # Temporal Transformer with Cross-Attention and Hierarchical Encoding
        x = self.temporal_transformer(x)

        # Layer Norm
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



