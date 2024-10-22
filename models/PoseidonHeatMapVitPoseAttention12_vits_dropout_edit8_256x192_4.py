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

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, temporal_depth):
        super(SpatioTemporalPatchEmbedding, self).__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_depth, patch_size, patch_size),
            stride=(temporal_depth, patch_size, patch_size),
            padding=(0, 0, 0)
        )

    def forward(self, x):
        # x shape: [batch_size, C, T, H, W]
        x = self.proj(x)  # [batch_size, embed_dim, T', H', W']
        batch_size, embed_dim, T_prime, H_prime, W_prime = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [batch_size, T', H', W', embed_dim]
        x = x.reshape(batch_size, -1, embed_dim)  # [batch_size, num_patches, embed_dim]
        return x

class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(SpatioTemporalPositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x shape: [batch_size, num_patches, embed_dim]
        x = x + self.pos_embed
        return x

class MotionExcitation(nn.Module):
    def __init__(self, embed_dim):
        super(MotionExcitation, self).__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: [batch_size, num_patches, embed_dim]
        # Compute motion features (differences between consecutive tokens)
        motion_features = x[:, 1:, :] - x[:, :-1, :]
        # Pad the last motion feature
        motion_features = F.pad(motion_features, (0, 0, 0, 1), mode='constant', value=0)
        excitation = torch.sigmoid(self.fc(motion_features))
        x = x * excitation
        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, dropout=0.1):
        super(SpatioTemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x shape: [batch_size, num_patches, embed_dim]
        x = self.transformer_encoder(x)
        return x

class PosePredictionHead(nn.Module):
    def __init__(self, embed_dim, num_joints, deconv_layer, final_layer):
        super(PosePredictionHead, self).__init__()
        self.deconv_layer = deconv_layer
        self.final_layer = final_layer
        # Optionally, you can include additional layers if needed

    def forward(self, x):
        # x shape: [batch_size, embed_dim, H', W']
        x = self.deconv_layer(x)
        x = self.final_layer(x)
        return x

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=8, num_layers=1, patch_size=16, temporal_depth=2):
        super(Poseidon, self).__init__()

        self.device = device
        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        
        self.backbone = self.model.backbone

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

        # Model parameters
        self.num_frames = cfg.WINDOWS_SIZE  # Total number of frames in the input sequence
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # e.g., [64, 48]
        self.embed_dim = cfg.MODEL.EMBED_DIM  # e.g., 768
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS  # e.g., 17
        self.patch_size = patch_size  # e.g., 16
        self.temporal_depth = temporal_depth  # e.g., 2
        self.num_layers = num_layers
        self.is_train = phase == 'train'

        # Spatio-Temporal Patch Embedding
        self.patch_embed = SpatioTemporalPatchEmbedding(
            in_channels=3,  # Assuming RGB images
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            temporal_depth=self.temporal_depth
        )

        # Calculate number of patches
        # Assuming input image size H x W, after patch embedding, the dimensions are reduced
        # Need to calculate the number of patches for positional encoding
        # For simplicity, assume input size is divisible by patch size
        input_size = cfg.MODEL.IMAGE_SIZE  # e.g., [256, 192]
        H, W = input_size
        T_prime = (self.num_frames - (self.temporal_depth - 1)) // self.temporal_depth
        H_prime = H // self.patch_size
        W_prime = W // self.patch_size
        num_patches = T_prime * H_prime * W_prime

        # Spatio-Temporal Positional Encoding
        self.pos_embed = SpatioTemporalPositionalEncoding(num_patches=num_patches, embed_dim=self.embed_dim)

        # Motion Excitation Module
        self.motion_excitation = MotionExcitation(embed_dim=self.embed_dim)

        # Spatio-Temporal Transformer
        self.transformer = SpatioTemporalTransformer(
            embed_dim=self.embed_dim,
            depth=self.num_layers,
            num_heads=self.num_heads,
            dropout=0.1
        )

        # Pose Prediction Head
        self.pose_head = PosePredictionHead(
            embed_dim=self.embed_dim,
            num_joints=self.num_joints,
            deconv_layer=self.deconv_layer,
            final_layer=self.final_layer
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M")
        print(f"Poseidon total parameters: {round(self.count_parameters() / 1e6, 1)} M")

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, meta=None):
        # x shape: [batch_size, num_frames, C, H, W]
        batch_size, num_frames, C, H, W = x.shape

        # Ensure num_frames matches
        assert num_frames == self.num_frames, f"Expected {self.num_frames} frames, but got {num_frames}"

        # Prepare input for patch embedding
        # Reshape x to [batch_size, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, C, num_frames, H, W]

        # Apply spatio-temporal patch embedding
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]

        # Apply positional encoding
        x = self.pos_embed(x)  # [batch_size, num_patches, embed_dim]

        # Apply motion excitation
        x = self.motion_excitation(x)  # [batch_size, num_patches, embed_dim]

        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply spatio-temporal transformer
        x = self.transformer(x)  # [batch_size, num_patches, embed_dim]

        # Reshape for the pose prediction head
        # First, reshape x back to spatial dimensions
        batch_size, num_patches, embed_dim = x.shape
        T_prime = (self.num_frames - (self.temporal_depth - 1)) // self.temporal_depth
        H_prime = H // self.patch_size
        W_prime = W // self.patch_size
        x = x.view(batch_size, T_prime, H_prime, W_prime, embed_dim)
        # For pose estimation, focus on the central time frame
        central_frame = T_prime // 2
        x = x[:, central_frame, :, :, :]  # [batch_size, H_prime, W_prime, embed_dim]
        x = x.permute(0, 3, 1, 2)  # [batch_size, embed_dim, H_prime, W_prime]

        # Upsample features if necessary to match deconv_layer input size
        expected_input_size = self.deconv_layer[0].in_channels
        if x.shape[2] != expected_input_size or x.shape[3] != expected_input_size:
            x = F.interpolate(x, size=(expected_input_size, expected_input_size), mode='bilinear', align_corners=False)

        # Apply pose prediction head
        x = self.pose_head(x)  # [batch_size, num_joints, heatmap_height, heatmap_width]

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



