import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.abspath('/home/pace/Poseidon/'))

import torch
import torch.nn as nn
import numpy as np
from datasets.zoo.posetrack.PoseTrack import PoseTrack
from posetimation import get_cfg, update_config 
from engine.defaults import default_parse_args
import torchvision.models as models
from torch.utils.data import DataLoader
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict
from mmpose.apis import init_model


class DummyModel(nn.Module):
    def __init__(self, device='cpu'):
        super(DummyModel, self).__init__()
        config_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288.py'
        checkpoint_file = '/home/pace/Poseidon/models/resnet/td-hm_res50_8xb64-210e_coco-384x288-7b8db90e_20220923.pth'
        self.device = device
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone

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

        return backbone_outputs


# Create a dummy input tensor
dummy_input = torch.randn(16, 5, 3, 192 , 256)  # (batch, num_frames, channels, height, width)

model = DummyModel(device="cuda:1")

# Forward pass
output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")