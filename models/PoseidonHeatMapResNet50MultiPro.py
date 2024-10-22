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
from torchvision.models._utils import IntermediateLayerGetter
import math


class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim_for_joint=30):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288.py'
        checkpoint_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288-7b8db90e_20220923.pth'
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone

        return_layers = {'layer3': 'layer3', 'layer4': 'layer4'}

        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=return_layers)

        # Define efficient convolutional layers for each backbone output
        self.layer_processors = nn.ModuleDict({
            'layer3': nn.Conv2d(1024, 512, kernel_size=1),
            'layer4': nn.Conv2d(2048, 512, kernel_size=1)
        })

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simplified attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
            
        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.output_sizes = 2048
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.embed_dim_for_joint = embed_dim_for_joint
        #self.embed_dim = self.num_joints * self.embed_dim_for_joint
        
        # Cross-attention for temporal information
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=self.num_heads, batch_first=True)

        # Deconv layers for heatmap generation
        self.deconv_layers = self._make_deconv_layers()
        
        # Final predictor layer
        self.final_layer = nn.Conv2d(in_channels=32, out_channels=self.num_joints, kernel_size=1, stride=1, padding=0)

        print(f"Balanced Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M")

                # check number of parameters of deconv layers
        print(f"Deconv layers parameters: {round(sum(p.numel() for p in self.deconv_layers.parameters()) / 1e6, 1)} M\n\n")


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        backbone_outputs = self.backbone(x)  # shape: [batch_size*num_frames, 384, 24, 18]

        # print("Backbone outputs:", backbone_outputs.shape) # torch.Size([batch_size*num_frames, 384, 24, 18])

        processed_outputs = []
        for layer_name, layer_output in backbone_outputs.items():
            processed_output = self.layer_processors[layer_name](layer_output)
            processed_output = self.global_pool(processed_output).squeeze(-1).squeeze(-1)

            # print("Processed output:", processed_output.shape) # torch.Size([batch_size*num_frames, 2048])
            processed_outputs.append(processed_output)

        # Stack processed outputs
        stacked_outputs = torch.stack(processed_outputs, dim=1)  # shape: [batch_size*num_frames, 4, 2048]

        # Calculate feature-level attention weights
        feature_weights = self.attention(stacked_outputs.view(-1, 2048)).view(batch_size*num_frames, 4, 2048)

        # Apply weighted sum
        weighted_outputs = stacked_outputs * feature_weights
        x = torch.sum(weighted_outputs, dim=1)  # shape: [batch_size*num_frames, 2048]

        # print("Weighted outputs:", x.shape) # torch.Size([batch_size*num_frames, 2048])

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
        input_channels = self.output_sizes
        upsample_configs = [
            (256, 2),
            (128, 2),
            (64, 2),
            (32, 2),
            (32, 2),
            (self.num_joints, 3)
        ]
        
        for out_channels, scale_factor in upsample_configs:
            layers.append(nn.ConvTranspose2d(
                input_channels, out_channels,
                kernel_size=scale_factor * 2,
                stride=scale_factor,
                padding=scale_factor // 2,
                output_padding=scale_factor % 2
            ))
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

