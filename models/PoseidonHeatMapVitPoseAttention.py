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
        self.attention = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        
        # Print number of parameters
        print(f"Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M\n\n")


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone
        x = self.backbone(x)[0]

        #print("Shape after backbone: ", x.shape) # torch.Size([batch_size*num_frames, 384, 24, 18])

        # Reshape to separate frames
        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18)

        # Compute attention weights
        # Average pooling over spatial dimensions
        x_pooled = self.pooling(x)

        #print("Shape after pooling: ", x_pooled.shape) # torch.Size([batch_size, num_frames, 384])

        x_pooled = x_pooled.view(batch_size, num_frames, self.embed_dim)

        print("Shape after reshaping: ", x_pooled.shape) # torch.Size([batch_size, num_frames, 384])

        attention_weights = self.attention(x_pooled).squeeze(-1)  # [batch_size, num_frames]

        #print("Shape of attention weights: ", attention_weights.shape) # torch.Size([batch_size, num_frames])

        attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        #print("Shape of attention weights after softmax: ", attention_weights.shape) # torch.Size([batch_size, num_frames, 1, 1, 1])

        # Apply attention weights
        x_weighted = x * attention_weights

        #print("Shape of x_weighted: ", x_weighted.shape) # torch.Size([batch_size, num_frames, 384, 24, 18])
        
        # Sum across frames
        x = x_weighted.sum(dim=1)  # [batch_size, embed_dim, 24, 18]

        #print("Shape after attention: ", x.shape) # torch.Size([batch_size, 384, 24, 18])

        # Deconvolution layers
        x = self.deconv_layer(x)

        #print("Shape after deconvolution: ", x.shape) # torch.Size([batch_size, 17, 96, 72])

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



