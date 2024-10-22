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
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase=TRAIN_PHASE, num_heads=5, embed_dim_for_joint=30):
        super(Poseidon, self).__init__()
        self.device = device
        self.backbone_model = Backbones(cfg, self.device)
        self.backbone = self.backbone_model.backbone.to(self.device)

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)

        # Get model info
        num_params, output_sizes = self.backbone_model.get_model_info()
        print(f"Backbone parameters: {round(num_params/1e6,1)} M, Output sizes: {output_sizes}")

        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.embed_dim_for_joint = embed_dim_for_joint
        self.embed_dim = self.num_joints * self.embed_dim_for_joint

        # Ensure embed_dim is divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # Create separate FC layers for each backbone output
        self.fc_layers = nn.ModuleDict()
        for layer, size in output_sizes.items():
            # Add adaptive pooling to reduce spatial dimensions
            self.fc_layers[layer] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(size, self.embed_dim),
                nn.Dropout(p=0.5)
            )

        # Self-attention layer for joints
        # self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim_for_joint, num_heads=self.num_heads, batch_first=True)

        # Cross-attention layer for frames
        self.cross_attention = nn.Sequential(
            nn.MultiheadAttention(embed_dim=self.embed_dim*4, num_heads=self.num_heads, batch_first=True),
            nn.Dropout(p=0.1)  # Add dropout with 10% probability
        )

        # Deconv layers for heatmap generation
        self.deconv_layers = self._make_deconv_layers()

        # Final predictor layer
        self.final_layer = nn.Conv2d(in_channels=self.num_joints, out_channels=self.num_joints, kernel_size=1, stride=1, padding=0)

        self.is_train = True if phase == TRAIN_PHASE else False

        # Print number of parameters
        print(f"Poseidon parameters: {round(self.number_of_parameters(), 1)} M\n\n")

    def _make_deconv_layers(self):
        layers = []
        input_channels = self.embed_dim*4
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

        # Replace final (1, 25) convolution with adaptive average pooling
        layers.append(nn.AdaptiveAvgPool2d((96, 72)))

        return nn.Sequential(*layers)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape

        x = x.view(-1, C, H, W)
        backbone_outputs = self.backbone(x)

        # Process each backbone output
        processed_features = []
        for layer, feature in backbone_outputs.items():
            fc_output = self.fc_layers[layer](feature)
            processed_features.append(fc_output)

        # Combine processed features with concatenation
        x = torch.cat(processed_features, dim=1)

        x = x.view(batch_size, num_frames, self.num_joints, self.embed_dim_for_joint*4)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, num_frames, self.embed_dim*4)
        # print(f"Shape after self-attention: {x.shape}")

        central_frame = x[:, num_frames // 2, :].unsqueeze(1)
        context_frames = x
        x, _ = self.cross_attention[0](central_frame, context_frames, context_frames)
        x = self.cross_attention[1](x)  # Apply dropout
        x = x.squeeze(1)
        x = x.view(batch_size, self.embed_dim*4, 1, 1)
        # print(f"Before deconv layers: {x.shape}")

        x = self.deconv_layers(x)
        # print(f"After deconv layers: {x.shape}")

        heatmaps = self.final_layer(x)
        # print(f"Final heatmaps shape: {heatmaps.shape}")

        return heatmaps
        

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


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
