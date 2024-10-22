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
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=384):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py'
        checkpoint_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth'

        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

        # Scongelamento parziale del backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.backbone.ln1.parameters():
            param.requires_grad = True

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.embed_dim = embed_dim # 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS # 17
        
        self.is_train = True if phase == 'train' else False


        # Attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.embed_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        
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

        # Backbone
        x = self.backbone(x)[0]
        # Reshape to separate frames
        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18)

        # Compute channel attention weights
        x_channel_pooled = self.pooling(x).view(batch_size, num_frames, self.embed_dim)
        channel_weights = self.channel_attention(x_channel_pooled).squeeze(-1)
        channel_weights = F.softmax(channel_weights, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Compute spatial attention weights
        x_spatial = x.view(-1, self.embed_dim, 24, 18)
        spatial_weights = self.spatial_attention(x_spatial)
        spatial_weights = spatial_weights.view(batch_size, num_frames, 1, 24, 18)
        spatial_weights = F.softmax(spatial_weights.view(batch_size, num_frames, -1), dim=-1).view(batch_size, num_frames, 1, 24, 18)

        # Combine channel and spatial attention
        combined_weights = channel_weights * spatial_weights

        # Apply combined attention weights
        x_weighted = x * combined_weights

        # Sum across frames
        x = x_weighted.sum(dim=1)  # [batch_size, embed_dim, 24, 18]

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



