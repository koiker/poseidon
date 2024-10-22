import os
import sys
import torch
import torch.nn as nn
import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict
from .backbones import Backbones

from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE


class SimpleBaseline(nn.Module):
    def __init__(self, cfg, device='cpu', phase=TRAIN_PHASE, num_heads=5, embed_dim_for_joint=30):
        super(SimpleBaseline, self).__init__()
        self.device = device
        self.backbone_model = Backbones(cfg, self.device)
        self.backbone = self.backbone_model.backbone.to(self.device)
        

        # Get model info
        num_params, output_size = self.backbone_model.get_model_info()
        print(f"Backbone parameters: {round(num_params/1e6,1)} M, Output size: {output_size}")
        
        self.num_heads = num_heads

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.embed_dim_for_joint = embed_dim_for_joint
        self.embed_dim = self.num_joints * self.embed_dim_for_joint

        # Ensure embed_dim is divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # Fully connected layer for processing backbone output
        self.fc = nn.Linear(output_size, self.embed_dim)

        # Output layer to map embeddings to keypoint coordinates
        self.keypoint_output = nn.Sequential(
            nn.LayerNorm(self.embed_dim_for_joint),
            nn.Linear(self.embed_dim_for_joint, 2)  # Assuming 2D keypoints (x, y)
        )

        self.sigma_output = nn.Sequential(
            nn.LayerNorm(self.embed_dim_for_joint),
            nn.Linear(self.embed_dim_for_joint, 1)  # Assuming 2D keypoints (x, y)
        )

        self.is_train = True if phase == TRAIN_PHASE else False


        self.joint_ordor =  [1,2,0,0,0,3,6,4,7,5,8,9,12,10,13,11,14]


        # print number of parameters
        print(f"Simple Baseline parameters: {round(self.number_of_parameters(), 1)} M\n\n")

    def forward(self, x, meta=None):
        batch_size, num_images, C, H, W = x.shape

        # Ensure there are 3 images in the input
        if num_images != 3:
            raise ValueError(f"Expected 3 images in the input, but got {num_images} images instead.")

        # Select the central image (input_x)
        x = x[:, 1, :, :, :]  # (batch_size, C, H, W)

        # Check if the number of channels is correct
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels in the input, but got {x.shape[1]} channels instead.")

        # Process the central image through the backbone
        processed_video = self.backbone(x)  # (batch_size, output_size)
        print(f"Processed video shape: {processed_video.shape}")

        # Apply fully connected layer
        x = self.fc(processed_video)  # (batch_size, self.embed_dim)
        print(f"Shape after fc layer: {x.shape}")

        # Reshape x to match the expected input shape for LayerNorm
        x = x.view(batch_size, self.num_joints, self.embed_dim_for_joint)  # (batch_size, num_joints, embed_dim_for_joint)
        print(f"Shape after reshaping for LayerNorm: {x.shape}")

        # Final output layer to predict keypoints
        keypoints = self.keypoint_output(x)  # (batch_size, self.num_joints, 2)
        print(f"Keypoints shape: {keypoints.shape}")

        # Final output layer to predict sigma
        sigma = self.sigma_output(x).sigmoid()  # (batch_size, self.num_joints, 1)
        print(f"Sigma shape: {sigma.shape}")

        score = 1 - sigma
        score = torch.mean(score, dim=2, keepdim=True)
        print(f"Score shape: {score.shape}")

        if self.is_train:
            target = meta['target'].to(self.device)
            target_weight = meta['target_weight'].to(self.device)
            acc = self.get_accuracy(keypoints, target, target_weight)
        else:
            acc = None

        output = EasyDict(
            pred_jts=keypoints,
            sigma=sigma,
            maxvals=score.float(),
            acc=acc
        )

        return output

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
