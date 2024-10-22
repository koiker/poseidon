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

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x




class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=512):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288.py'
        checkpoint_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288-7b8db90e_20220923.pth'
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone

        return_layers = {'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=return_layers)
            
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.embed_dim = embed_dim

        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear embeddings for each feature layer
        self.embed_layer1 = nn.Linear(256, embed_dim)
        self.embed_layer2 = nn.Linear(512, embed_dim)
        self.embed_layer3 = nn.Linear(1024, embed_dim)
        self.embed_layer4 = nn.Linear(2048, embed_dim)

        # Attention weights for feature layers
        self.attention_weights = nn.Parameter(torch.ones(4, embed_dim) / 4)
        self.softmax = nn.Softmax(dim=0)

        # Positional encoding
        self.position_encoding = PositionalEncoding(embed_dim)

        # Multiple Cross-attention layers with 3 cross-attention layer and 3 normalization layers
        self.cross_attention1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention3 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Deconv layers for heatmap generation
        self.deconv_layers = self._make_deconv_layers()
        
        # Final predictor layer
        self.final_layer = nn.Conv2d(in_channels=self.num_joints, out_channels=self.num_joints, kernel_size=1, stride=1, padding=0)
        
        print(f"Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M\n")
        print(f"Deconv layers parameters: {round(sum(p.numel() for p in self.deconv_layers.parameters()) / 1e6, 1)} M\n")

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        backbone_outputs = self.backbone(x)

        # Process each layer
        layer1 = self.adaptive_pool(backbone_outputs['layer1'])
        layer2 = self.adaptive_pool(backbone_outputs['layer2'])
        layer3 = self.adaptive_pool(backbone_outputs['layer3'])
        layer4 = self.adaptive_pool(backbone_outputs['layer4'])

        # Apply linear embeddings
        layer1 = self.embed_layer1(layer1.view(batch_size * num_frames, -1)) 
        layer2 = self.embed_layer2(layer2.view(batch_size * num_frames, -1))
        layer3 = self.embed_layer3(layer3.view(batch_size * num_frames, -1))
        layer4 = self.embed_layer4(layer4.view(batch_size * num_frames, -1))

        # Stack the embedded layers
        stacked_layers = torch.stack([layer1, layer2, layer3, layer4], dim=1)


        # Apply attention weights
        attention_weights = self.softmax(self.attention_weights)  # Shape: (4, embed_dim)
        x = torch.sum(stacked_layers * attention_weights, dim=1)

        # Reshape the embeddings
        x = x.view(batch_size, num_frames, -1)

        # Add positional encoding
        x = self.position_encoding(x)

        # Get the central frame and context frames
        central_frame = x[:, num_frames // 2, :].unsqueeze(1)
        context_frames = x
        
        # Apply cross-attention layers
        x, _ = self.cross_attention1(central_frame, context_frames, context_frames)
        x = self.norm1(x)

        x, _ = self.cross_attention2(x, context_frames, context_frames)
        x = self.norm2(x)

        x, _ = self.cross_attention3(x, context_frames, context_frames)
        x = self.norm3(x)

        # Apply deconv layers
        x = x.view(batch_size, -1, 1, 1) 
        x = self.deconv_layers(x)

        heatmap = self.final_layer(x)

        return heatmap

    def _make_deconv_layers(self):
        layers = []
        input_channels = self.embed_dim  # Adjusted for the concatenated embeddings
        upsample_configs = [
            (256, 2),
            (128, 2),
            (64, 2),
            (32, 2),
            (32, 2),
            (self.num_joints, 3)
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

