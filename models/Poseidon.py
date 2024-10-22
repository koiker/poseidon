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


class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase=TRAIN_PHASE, num_heads=5, embed_dim_for_joint=30):
        super(Poseidon, self).__init__()
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

        # Multi-head attention for cross-frame attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

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
        print(f"Poseidon parameters: {round(self.number_of_parameters(), 1)} M\n\n")

    def forward(self, x, meta=None):        

        batch_size, num_frames, C, H, W = x.shape

        # Backbone expects [batch_size * num_frames, C, H, W]
        x = x.view(-1, C, H, W)

        # Process video frames through the backbone
        processed_video = self.backbone(x) # [batch_size * num_frames, output_size]

        # Apply fully connected layer
        x = self.fc(processed_video) # [batch_size * num_frames, embed_dim]

        # Reshape for attention: [num_frames, batch_size, embed_dim]
        x = x.view(batch_size, num_frames, -1).permute(1, 0, 2) # [num_frames, batch_size, embed_dim]

        # Central frame as query, all frames as key and value
        central_frame = x[num_frames // 2].unsqueeze(0) # [1, batch_size, embed_dim]
        
        attn_output, _ = self.cross_attention(central_frame, x, x) # [1, batch_size, embed_dim]

        # Reshape back to [batch_size, num_joints, embed_dim_for_joint]
        attn_output = attn_output.permute(1, 0, 2).contiguous().view(batch_size, self.num_joints, self.embed_dim_for_joint) # [batch_size, num_joints, embed_dim_for_joint]

        # Reordering step
        idx = self.joint_ordor 
        attn_output = attn_output[:, idx]

        # Final output layer to predict keypoints
        keypoints = self.keypoint_output(attn_output) # [batch_size, num_joints, 2]
        
        # Final output layer to predict sigma
        sigma = self.sigma_output(attn_output).sigmoid() # [batch_size, num_joints, 1]

        score = 1 - sigma

        score = torch.mean(score, dim=2, keepdim=True) 

        target = meta['target'].to(self.device)
        target_weight = meta['target_weight'].to(self.device)

        acc = self.get_accuracy(keypoints, target, target_weight)

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

    def accuracy_heatmap(output, target, hm_type='gaussian', thr=0.5):
        '''
        Calculate accuracy according to PCK (),
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        '''
        idx = list(range(output.shape[1]))
        norm = 1.0
        if hm_type == 'gaussian':
            pred, _ = get_max_preds(output)
            target, _ = get_max_preds(target)
            h = output.shape[2]
            w = output.shape[3]
            norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        dists = calc_dists(pred, target, norm)  # use a fixed length as a measure rather than the length of body parts

        acc = np.zeros((len(idx) + 1))
        avg_acc = 0
        cnt = 0

        for i in range(len(idx)):
            acc[i + 1] = dist_acc(dists[idx[i]], thr)
            if acc[i + 1] >= 0:
                avg_acc = avg_acc + acc[i + 1]
                cnt += 1

        avg_acc = avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
            acc[0] = avg_acc

        return acc, avg_acc, cnt, pred

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
