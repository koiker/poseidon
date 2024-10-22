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
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288.py'
        checkpoint_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288-7b8db90e_20220923.pth'
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone
            
        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
                
        # Print number of parameters
        print(f"Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M\n\n")



    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        backbone_outputs = self.backbone(x)[0]  # shape: torch.Size([80, 2048, 12, 9])  

        print("Backbone outputs:", backbone_outputs.shape) # shape: torch.Size([80, 2048, 12, 9])  

        backbone_outputs = backbone_outputs.view(batch_size, num_frames, -1, self.heatmap_size[0], self.heatmap_size[1])

        print("Backbone outputs reshaped:", backbone_outputs.shape) # torch.Size([16, 5, 32, 72, 96])


        return attended_output

    def set_phase(self, phase):
        self.phase = phase
        self.is_train = True if phase == TRAIN_PHASE else False

    def get_phase(self):
        return self.phase



