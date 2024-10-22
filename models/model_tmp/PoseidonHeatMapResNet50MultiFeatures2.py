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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FeatureFusion(nn.Module):
    def __init__(self, in_channels1=1024, in_channels2=2048, out_channels=2048, mode='concat'):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels // 2, kernel_size=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mode = mode
        
        # Channel Attention modules
        self.attention1 = ChannelAttention(out_channels // 2)
        self.attention2 = ChannelAttention(out_channels // 2)
        
        # If using concat mode, we need attention for the concatenated feature
        if self.mode == 'concat':
            self.attention_concat = ChannelAttention(out_channels)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        x1 = self.downsample(x1)  # Downsample x1 to match x2's spatial dimensions
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        # Apply channel attention
        x1 = self.attention1(x1)
        x2 = self.attention2(x2)

        
        if self.mode == 'concat':
            output = torch.cat([x1, x2], dim=1)
            output = self.attention_concat(output)
            return self.global_pool(output)
        elif self.mode == 'add':
            return x1 + x2
        else:
            raise ValueError("Mode must be either 'concat' or 'add'")
class PixelShuffleDeconv(nn.Module):
    def __init__(self, input_channels, output_channels, upscale_factor):
        super(PixelShuffleDeconv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels * upscale_factor**2, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeconvNetwork(nn.Module):
    def __init__(self, input_channels=2048, num_joints=17):
        super(DeconvNetwork, self).__init__()
        self.input_channels = input_channels
        self.num_joints = num_joints

        self.deconv_layers = self.make_deconv_layers()

    def make_deconv_layers(self):
        layers = []
        input_channels = self.input_channels
        upsample_configs = [
            (512, 2),  # [1, 1] -> [2, 2]
            (256, 2),  # [2, 2] -> [4, 4]
            (128, 2),  # [4, 4] -> [8, 8]
            (64, 2),   # [8, 8] -> [16, 16]
            (32, 2),   # [16, 16] -> [32, 32]
        ]

        for out_channels, scale_factor in upsample_configs:
            layers.append(PixelShuffleDeconv(input_channels, out_channels, scale_factor))
            input_channels = out_channels

        # Final layer to get to num_joints channels
        layers.append(nn.Conv2d(input_channels, self.num_joints, kernel_size=1))
        
        # Upsample to 96x96
        layers.append(nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False))

        # Add the final pooling layer to get 96x72 output
        layers.append(nn.AdaptiveAvgPool2d((96, 72)))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288.py'
        checkpoint_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288-7b8db90e_20220923.pth'
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone

        return_layers = {'layer3': 'layer3', 'layer4': 'layer4'}

        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=return_layers)
            
        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.output_sizes = 2048
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS

        self.feature_fusion = FeatureFusion(mode='concat')
        
        # cross-attention layer for frames
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.output_sizes, num_heads=self.num_heads, batch_first=True)

        # Deconvolutional layers
        self.deconv_layers = DeconvNetwork(input_channels=self.output_sizes, num_joints=self.num_joints)
        
        # Final predictor layer
        self.final_layer = nn.Conv2d(in_channels=self.num_joints, out_channels=self.num_joints, kernel_size=1, stride=1, padding=0)
        
        # Print number of parameters
        print(f"Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M\n\n")

        self.deconv_layers = DeconvNetwork(input_channels=self.output_sizes, num_joints=self.num_joints)

                # check number of parameters of deconv layers
        print(f"Deconv layers parameters: {round(sum(p.numel() for p in self.deconv_layers.parameters()) / 1e6, 1)} M\n\n")



    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        backbone_outputs = self.backbone(x)

        # get output 
        layer3, layer4 = backbone_outputs['layer3'], backbone_outputs['layer4']

        # Apply feature fusion
        backbone_outputs = self.feature_fusion(layer3, layer4)  # shape: [batch_size*num_frames, 2048, 1, 1]

        x = backbone_outputs.view(batch_size, num_frames, -1)  # shape: [batch_size, num_frames, 384]

        central_frame = x[:, num_frames // 2, :].unsqueeze(1)  # shape: [batch_size, 1, 384]

        # print("Central frame:", central_frame.shape) # torch.Size([batch_size, 1, 384])

        context_frames = x # shape: [batch_size, num_frames, 384]

        # print("Context frames:", context_frames.shape) # torch.Size([batch_size, num_frames, 384])

        # Apply cross-attention
        x, _ = self.cross_attention(central_frame, context_frames, context_frames)  # shape: [batch_size, 1, 384]

        # print("After cross-attention:", x.shape) # torch.Size([batch_size, 1, 384])

        # Apply deconv layers
        x = x.view(batch_size, -1, 1, 1)  # shape: [batch_size, 384, 1, 1]
        
        # Deconvolutional layers
        x = self.deconv_layers(x)

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

