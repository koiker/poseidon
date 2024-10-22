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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=6, embed_dim_for_joint=30):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py'
        checkpoint_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth'

        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone
        

        # print(self.backbone) # torch.Size([batch_size*num_frames, 384, 24, 18])

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
        self.output_sizes = 368
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.embed_dim_for_joint = embed_dim_for_joint
        self.embed_dim = self.num_joints * self.embed_dim_for_joint
        
        # Ensure embed_dim is divisible by num_heads
        # assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(self.output_sizes)

        # Final pooling
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Global context block
        self.global_context = nn.Sequential(
            nn.Conv2d(self.output_sizes, self.output_sizes, kernel_size=1),
            nn.LayerNorm([self.output_sizes, 24, 18]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_sizes, self.output_sizes, kernel_size=1),
            nn.Sigmoid()
        )

        # cross-attention layer for frames
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.output_sizes, num_heads=self.num_heads, batch_first=True)

        # Deconv layers for heatmap generation
        self.deconv_layers = self._make_deconv_layers()
        
        # Final predictor layer
        self.final_layer = nn.Conv2d(in_channels=self.num_joints, out_channels=self.num_joints, kernel_size=1, stride=1, padding=0)
        
        self.is_train = True if phase == 'train' else False
        
        # Print number of parameters
        print(f"Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M\n\n")


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        backbone_outputs = self.backbone(x)[0]  # shape: [batch_size*num_frames, 384, 24, 18]

        # Apply spatial attention
        spatial_weights = self.spatial_attention(backbone_outputs)
        x = backbone_outputs * spatial_weights

        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # Apply global context
        context = self.global_context(x)
        x = x * context

        # Final pooling
        x = self.final_pool(x)  # shape: [batch_size*num_frames, 384, 1, 1]

        x = x.view(batch_size, num_frames, -1)  # shape: [batch_size, num_frames, 384]

        central_frame = x[:, num_frames // 2, :].unsqueeze(1)  # shape: [batch_size, 1, 384]

        # print("Central frame:", central_frame.shape) # torch.Size([batch_size, 1, 384])

        context_frames = x # shape: [batch_size, num_frames, 384]

        # print("Context frames:", context_frames.shape) # torch.Size([batch_size, num_frames, 384])

        # Apply cross-attention
        x, _ = self.cross_attention(central_frame, context_frames, context_frames)  # shape: [batch_size, 1, 384]

        # print("After cross-attention:", x.shape) # torch.Size([batch_size, 1, 384])

        # Apply deconv layers
        x = x.view(batch_size, -1, 1, 1)  # shape: [batch_size, 384, 1, 1]
        x = self.deconv_layers(x)  # shape: [batch_size, 17, 96, 72]

        # print("After deconv layers:", x.shape) # torch.Size([batch_size, 17, 96, 72])

        heatmap = self.final_layer(x)  # shape: [batch_size, 17, 96, 72]

        return heatmap

    def _make_deconv_layers(self):
        layers = []
        input_channels = self.output_sizes  # Adjusted for the flattened input
        upsample_configs = [
            (256, 2),  # [1, 1] -> [2, 2]
            (128, 2),  # [2, 2] -> [4, 4]
            (64, 2),   # [4, 4] -> [8, 8]
            (32, 2),   # [8, 8] -> [16, 16]
            (32, 2),   # [16, 16] -> [32, 32]
            (self.num_joints, 3)  # [32, 32] -> [96, 96]
        ]
        
        for out_channels, scale_factor in upsample_configs:
            layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, bias=False))
            if out_channels != self.num_joints:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            input_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d((96, 72)))
        return nn.Sequential(*layers)

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



