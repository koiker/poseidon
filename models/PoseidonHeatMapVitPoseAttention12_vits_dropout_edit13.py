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

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', embed_dim=384):
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
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_frames = cfg.WINDOWS_SIZE
        self.is_train = True if phase == 'train' else False

        # Attention-based weighted sum
        self.attention = nn.Sequential(
            nn.Conv2d(self.embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Adding dropout with 30% rate after ReLU
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Adding dropout after second ReLU
            nn.Conv2d(64, 1, kernel_size=1)  # Output a single score per frame
        )

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

        # Compute attention scores for each frame
        attention_scores = []
        for frame in range(num_frames):
            attention_score = self.attention(x[:, frame])  # Shape: [batch_size, 1, 24, 18]
            attention_scores.append(attention_score)
        attention_scores = torch.stack(attention_scores, dim=1)  # Shape: [batch_size, num_frames, 1, 24, 18]

        # Normalize attention scores using softmax across the frames
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, num_frames, 1, 24, 18]

        # Apply attention weights to the frames
        x_weighted = x * attention_weights  # Element-wise multiplication

        # Sum over frames to aggregate features
        x = x_weighted.sum(dim=1)  # Shape: [batch_size, embed_dim, 24, 18]

        # Layer Norm
        x = self.layer_norm(x)

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



