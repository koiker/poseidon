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

from torchvision.ops import DeformConv2d
from posetimation import get_cfg, update_config 

class DeformableConvolutionLayer(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super(DeformableConvolutionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = DeformConv2d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input shape: [batch_size, embed_dim, height, width]
        batch_size, channels, height, width = x.shape
        
        # Generate the correct offset tensor with shape [batch_size, 2 * kernel_size * kernel_size, height, width]
        # For a 3x3 kernel, the offset will have 18 channels
        offset_channels = 2 * self.kernel_size * self.kernel_size
        x_offset = torch.randn(batch_size, offset_channels, height, width, device=x.device, dtype=x.dtype)
        
        # Apply deformable convolution
        x = self.deform_conv(x, x_offset)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = torch.mean(x, dim=(2, 3))  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, channels, 1, 1)
        return x * y

class AdaptiveFrameWeighting(nn.Module):
    def __init__(self, embed_dim, num_frames):
        super(AdaptiveFrameWeighting, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        self.frame_quality_estimator = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, height, width]
        batch_size, num_frames, embed_dim, height, width = x.shape
        
        # Estimate quality for each frame
        x_reshaped = x.view(batch_size * num_frames, embed_dim, height, width)
        quality_scores = self.frame_quality_estimator(x_reshaped).view(batch_size, num_frames)
        
        # Normalize scores using sigmoid instead of softmax
        weights = torch.sigmoid(quality_scores).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Weight frames
        weighted_x = x * weights
        
        return weighted_x, weights.squeeze()

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(0.1)


    def forward(self, query, context):
        # Reshape input: [B, C, H, W] -> [H*W, B, C]
        B, num_context_frames, C, H, W = context.shape
        query = query.view(B, C, -1).permute(2, 0, 1) # [H*W, B, C]
        context = context.view(B, num_context_frames, C, -1).permute(3, 0, 1, 2) # [H*W, B, num_context_frames, C]
        context = context.reshape(-1, B, C) # [H*W*num_context_frames, B, C]

        # Apply weighted attention
        attn_output, _ = self.mha(query, context, context)
        attn_output = self.dropout(attn_output)

        # Reshape output: [H*W, B, C] -> [B, C, H, W]
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        return attn_output


class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=384):
        super(Poseidon, self).__init__()
        
        self.device = device
        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        
        self.backbone = self.model.backbone

        self.return_layers = {'layers.3': 'layer3', 'layer.7': 'layer7',}
        #self.return_layers = {'layers.9': 'layer9', 'layer.21': 'layer21',}

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE

        # Scongelamento parziale del backbone
        if cfg.MODEL.FREEZE_WEIGHTS:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for layer in self.backbone.layers[-12:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.backbone.ln1.parameters():
                param.requires_grad = True

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.embed_dim = embed_dim # 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS # 17
        self.num_frames = cfg.WINDOWS_SIZE
        
        self.is_train = True if phase == 'train' else False

        # Adaptive Frame Weighting
        self.adaptive_weighting = AdaptiveFrameWeighting(self.embed_dim, self.num_frames)

        # Cross-Attention
        self.cross_attention = CrossAttention(self.embed_dim, self.num_heads)

        # Deformable convolution after backbone feature extraction
        self.deformable_conv_layer = DeformableConvolutionLayer(embed_dim)

        # SE Block after deformable convolutions
        self.se_block = SEBlock(embed_dim)

        # Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)

        self.dropout = nn.Dropout(0.1)

        # Layer norms for intermediate features
        self.intermediate_layer_norms = nn.ModuleDict({
            name: self.backbone.ln1 if name == 'layer11' else nn.LayerNorm(embed_dim)
            for name in self.return_layers.values()
        })

        # Layer normalization
        self.layer_norm = nn.LayerNorm([embed_dim, 24, 18])

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")

        print(f"Poseidon parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone feature extraction
        model_output = self.backbone(x)[0]

        # deformable convolution
        model_output = self.deformable_conv_layer(model_output)

        # SE Block
        model_output = self.se_block(model_output)
        
        # Reshape to separate frames
        x = model_output.reshape(batch_size,num_frames, self.embed_dim, 24, 18) # [batch_size, num_frames, 384, 24, 18]  

        # Adaptive Frame Weighting
        x, frame_weights = self.adaptive_weighting(x)

        x = self.dropout(x)

        # Cross-Attention
        center_frame_idx = num_frames // 2
        center_frame = x[:, center_frame_idx]
        context_frames = torch.cat([x[:, :center_frame_idx], x[:, center_frame_idx+1:]], dim=1)

        context_frames = context_frames.view(-1, self.embed_dim, 24*18).permute(2, 0, 1)
        context_frames, _ = self.self_attention(context_frames, context_frames, context_frames)
        context_frames = self.dropout(context_frames)
        context_frames = context_frames.permute(1, 2, 0).view(batch_size, num_frames-1, self.embed_dim, 24, 18)

        # Cross-Attention
        attended_features = self.cross_attention(center_frame, context_frames)
        attended_features = self.dropout(attended_features)
        
        # Layer Norm
        attended_features = self.layer_norm(attended_features)

        # residual connection
        attended_features += center_frame

        # Deconvolution layers
        x = self.deconv_layer(attended_features)

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



