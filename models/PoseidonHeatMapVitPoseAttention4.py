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

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        batch_size, num_frames, embed_dim, H, W = x.shape
        
        # Average pooling over spatial dimensions
        x_pooled = self.pooling(x.view(-1, embed_dim, H, W)).view(batch_size, num_frames, embed_dim)
        
        # Compute attention weights for each channel
        attention_weights = self.attention(x_pooled)
        attention_weights = attention_weights.view(batch_size, num_frames, embed_dim, 1, 1)
        
        # Apply softmax to get attention distribution
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        x_weighted = x * attention_weights
        
        return x_weighted



class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_attention_layers=5, embed_dim=384):
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
        self.num_joints = cfg.MODEL.NUM_JOINTS # 17
        self.num_attention_layers=num_attention_layers
        
        self.is_train = True if phase == 'train' else False

        # Cascaded Attention Layers
        self.attention_layers = nn.ModuleList([AttentionLayer(embed_dim) for _ in range(num_attention_layers)])

        # Frame Fusion Layer
        self.frame_fusion = nn.Linear(num_attention_layers * embed_dim, embed_dim)
        
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

        # Attention mechanism
        attention_outputs = []
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
            attention_outputs.append(x.sum(dim=1))
            #print("Shape of x_weighted: ", x.shape)
            
         # Concatenate outputs from all attention layers
        x_concat = torch.cat(attention_outputs, dim=1)  # [batch_size, num_layers * embed_dim, 24, 18]

        #print("Shape of x_concat: ", x_concat.shape)
        
        # Fuse frame information
        x = self.frame_fusion(x_concat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [batch_size, embed_dim, 24, 18]

        #print("Shape of x after frame fusion: ", x.shape)

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



