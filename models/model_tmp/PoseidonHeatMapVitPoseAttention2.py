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


class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=384, num_weights_per_frame=16):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py'
        checkpoint_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth'
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

        # Partial unfreezing of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.backbone.ln1.parameters():
            param.requires_grad = True

        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.embed_dim = embed_dim  # 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS  # 17
        self.is_train = phase == 'train'
        self.num_weights_per_frame = num_weights_per_frame

        # Complex attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_weights_per_frame)
        )
        
        # Temporal mixing network
        self.temporal_mix = nn.Sequential(
            nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        print(f"Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M\n\n")

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone
        x = self.backbone(x)[0]
        
        # Reshape to separate frames
        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18)

        # Compute complex attention weights
        x_pooled = self.pooling(x).view(batch_size, num_frames, self.embed_dim)
        attention_weights = self.attention(x_pooled)  # [batch_size, num_frames, num_weights_per_frame]
        
        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Expand attention weights to match spatial dimensions
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        attention_weights = attention_weights.expand(-1, -1, -1, 24, 18)

        # Apply attention weights
        x_weighted = x.unsqueeze(2) * attention_weights.unsqueeze(3)
        x_weighted = x_weighted.view(batch_size, num_frames * self.num_weights_per_frame, self.embed_dim, 24, 18)

        # Temporal mixing
        x_mixed = self.temporal_mix(x_weighted.transpose(1, 2)).transpose(1, 2)
        
        # Aggregate across temporal dimension
        x = x_mixed.sum(dim=1)  # [batch_size, embed_dim, 24, 18]

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



