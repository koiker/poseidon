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


from posetimation import get_cfg, update_config 


import torch
import torch.nn as nn


class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=384, num_layers=3):
        super(Poseidon, self).__init__()

        self.device = device
        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        self.backbone = self.model.backbone

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE

        # Partial freezing of the backbone
        if cfg.MODEL.FREEZE_WEIGHTS:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for layer in self.backbone.layers[-12:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.backbone.ln1.parameters():
                param.requires_grad = True

        # Model parameters
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.is_train = True if phase == 'train' else False


        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")
        print(f"Poseidon total parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


    def forward(self, x, meta=None):
        """
        x: Input tensor of shape (batch_size, num_frames, C, H, W)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape x to process all frames through the backbone
        x = x.view(-1, C, H, W)  # Shape: (batch_size * num_frames, C, H, W)

        # Backbone feature extraction
        x = self.backbone(x)[0]  # Shape: (batch_size * num_frames, feature_dim, h, w)

        # Get spatial dimensions
        feature_dim = x.shape[1]
        h, w = x.shape[2], x.shape[3]

        # Reshape x back to (batch_size, num_frames, feature_dim, h, w)
        x = x.view(batch_size, num_frames, feature_dim, h, w)

        # Flatten spatial dimensions
        x = x.view(batch_size, num_frames, feature_dim, h * w)  # Shape: (batch_size, num_frames, feature_dim, h*w)

        # Permute dimensions to prepare for attention
        x = x.permute(0, 3, 1, 2).contiguous()  # Shape: (batch_size, h*w, num_frames, feature_dim)

        # Merge batch_size and spatial dimensions
        x = x.view(batch_size * h * w, num_frames, feature_dim)  # Shape: (batch_size * h*w, num_frames, feature_dim)

        # Transpose for attention module: (num_frames, batch_size * h*w, feature_dim)
        x = x.permute(1, 0, 2).contiguous()

        # Apply attention over the temporal dimension
        attn_output, _ = self.attention(x, x, x)  # Shape: (num_frames, batch_size * h*w, feature_dim)

        # Extract features corresponding to the central frame
        center_frame = num_frames // 2
        x = attn_output[center_frame]  # Shape: (batch_size * h*w, feature_dim)

        # Reshape back to (batch_size, h*w, feature_dim)
        x = x.view(batch_size, h * w, feature_dim)

        # Reshape to (batch_size, feature_dim, h, w)
        x = x.permute(0, 2, 1).contiguous().view(batch_size, feature_dim, h, w)

        # Deconvolution layers
        x = self.deconv_layer(x)

        # Final layer to get pose heatmaps
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



