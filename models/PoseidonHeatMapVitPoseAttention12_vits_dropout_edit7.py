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
import math


from posetimation import get_cfg, update_config 

class SpatioTemporalWeighting(nn.Module):
    def __init__(self, embed_dim, num_frames, num_heads):
        super(SpatioTemporalWeighting, self).__init__()
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, num_frames, embed_dim, height, width = x.shape
        x = x.view(batch_size, num_frames, embed_dim, -1).permute(1, 0, 2, 3)  # Shape: [num_frames, batch, embed_dim, hw]
        attn_output, _ = self.temporal_attn(x, x, x)
        attn_output = self.norm(attn_output + x)  # Residual connection
        return attn_output.permute(1, 0, 2, 3).view(batch_size, num_frames, embed_dim, height, width)

class AdaptiveFrameWeighting(nn.Module):
    def __init__(self, embed_dim, num_frames):
        super(AdaptiveFrameWeighting, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        self.frame_quality_estimator = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, height, width]
        batch_size, num_frames, embed_dim, height, width = x.shape
        
        # Estimate quality for each frame
        x_reshaped = x.view(batch_size * num_frames, embed_dim, height, width)
        quality_scores = self.frame_quality_estimator(x_reshaped).view(batch_size, num_frames)
        
        # Normalize scores
        weights = F.softmax(quality_scores, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Weight frames
        weighted_x = x * weights
        
        return weighted_x, weights.squeeze()

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.paths = nn.ModuleList()
        for pool_size in pool_sizes:
            self.paths.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # Add a 1x1 convolution to adjust channel dimensions
        self.adjust_dim = nn.Conv2d(in_channels + out_channels * len(pool_sizes), in_channels, kernel_size=1)
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x]
        for path in self.paths:
            features.append(F.interpolate(path(x), size=(h, w), mode='bilinear', align_corners=True))
        out = self.adjust_dim(torch.cat(features, dim=1))
        return out

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiScaleFeatureFusion, self).__init__()
        self.ppm = PyramidPoolingModule(embed_dim, embed_dim // 4)
        self.fusion_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.fusion_norm = nn.BatchNorm2d(embed_dim)
        self.fusion_act = nn.ReLU(inplace=True)
        self.attention_fusion = AttentionFusion(embed_dim, num_heads)
        self.dropout = nn.Dropout(0.1)
        
        # Add scale-specific attention
        self.scale_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, features):
        multi_scale_features = {}
        for name, feature in features.items():
            B, num_frames, C, H, W = feature.shape
            feature = feature.view(-1, C, H, W)
            multi_scale_feature = self.ppm(feature)
            multi_scale_feature = self.fusion_conv(multi_scale_feature)
            multi_scale_feature = self.fusion_norm(multi_scale_feature)
            multi_scale_feature = self.fusion_act(multi_scale_feature)
            multi_scale_feature = self.dropout(multi_scale_feature)
            multi_scale_feature = multi_scale_feature.view(B, num_frames, C, H, W)
            multi_scale_features[name] = multi_scale_feature
        
        # Apply scale-specific attention
        scales = list(multi_scale_features.keys())
        scale_features = torch.stack([multi_scale_features[s] for s in scales], dim=0)
        scale_features = scale_features.view(len(scales), -1, C)
        attended_scales, _ = self.scale_attention(scale_features, scale_features, scale_features)
        attended_scales = attended_scales.view(len(scales), B, num_frames, C, H, W)
        
        for i, scale in enumerate(scales):
            multi_scale_features[scale] = attended_scales[i]
        
        # Use the existing AttentionFusion module
        fused_features = self.attention_fusion(multi_scale_features)
        return fused_features

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        # Assuming features is a dictionary of tensors, each with shape [10, 432, 384]
        B, num_frames, embed_dim, H, W = list(features.values())[0].shape

       
        # for each item in feature dict
        for key, value in features.items():
            features[key] = value.reshape(B*num_frames, H*W, embed_dim)

        # Concatenate features along the sequence dimension
        features_cat = torch.cat(list(features.values()), dim=1)  # Shape: [10, 432*num_features, 384]

        # get shape of features
        B_numframes, _, embed_dim = features_cat.shape
        
        # Transpose for attention: [432*num_features, 10, 384]
        features_cat = features_cat.transpose(0, 1)
        
        # Apply self-attention
        attn_output, _ = self.attention(features_cat, features_cat, features_cat)
        
        # Add residual connection and layer norm
        fused_features = self.norm(features_cat + attn_output)
        fused_features = self.dropout(fused_features)
        
        # Project back to original sequence length
        fused_features = fused_features.transpose(0, 1)  # Shape: [10, 432*num_features, 384]
        fused_features = fused_features.view(B_numframes, len(features), -1, embed_dim)  # Shape: [10, num_features, 432, 384]
        fused_features = torch.mean(fused_features, dim=1)  # Shape: [10, 432, 384]
        
        # Final projection to ensure we capture information from all feature levels
        fused_features = self.final_proj(fused_features)
        
        return fused_features


class ExtractIntermediateLayers(nn.Module):
    def __init__(self, model, return_layers):
        super(ExtractIntermediateLayers, self).__init__()
        self.model = model
        self.return_layers = return_layers
        self.features = {}

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for name, layer in self.return_layers.items():
            layer_idx = int(name.split('.')[1])
            self.model.layers[layer_idx].register_forward_hook(get_hook(layer))

    def forward(self, x):
        self.features = {}
        model_output = self.model(x)[0]
        return self.features, model_output

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

    def forward(self, query, context):
        B, num_context_frames, C, H, W = context.shape
        
        # Reshape and add positional encoding
        query = query.view(B, C, -1).permute(2, 0, 1)  # [H*W, B, C]
        query = self.positional_encoding(query)
        
        context = context.view(B, num_context_frames, C, -1).permute(3, 0, 1, 2)  # [H*W, B, num_context_frames, C]
        context = context.reshape(-1, B, C)  # [H*W*num_context_frames, B, C]
        context = self.positional_encoding(context)

        # Multi-head attention
        attn_output, _ = self.mha(query, context, context)
        attn_output = self.dropout(attn_output)
        output = self.norm1(query + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(output)
        ffn_output = self.dropout(ffn_output)
        output = self.norm2(output + ffn_output)

        # Reshape output: [H*W, B, C] -> [B, C, H, W]
        output = output.permute(1, 2, 0).view(B, C, H, W)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=384):
        super(Poseidon, self).__init__()
        
        self.device = device

        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        self.backbone = self.model.backbone

        #self.return_layers = {'layers.7': 'layer7','layers.15':'layers15', 'layers.23': 'layers23'}
        #self.return_layers = {'layers.9': 'layer9', 'layer.21': 'layer21',}
        self.return_layers = {'layers.3': 'layer3', 'layers.7': 'layer7'}

        self.extract_layers = ExtractIntermediateLayers(self.backbone, self.return_layers)

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE
   
        # Scongelamento parziale del backbone
        if cfg.MODEL.FREEZE_WEIGHTS:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for layer in self.backbone.layers[-12:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.backbone.ln1.parameters():
                param.requires_grad = True
                

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.embed_dim = embed_dim # 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS # 17
        self.num_frames = cfg.WINDOWS_SIZE
        
        self.is_train = True if phase == 'train' else False

        # Feature Fusion
        self.feature_fusion = MultiScaleFeatureFusion(self.embed_dim, num_heads=self.num_heads)

        # Adaptive Frame Weighting
        self.adaptive_weighting = AdaptiveFrameWeighting(self.embed_dim, self.num_frames)

        # Cross-Attention
        self.cross_attention = CrossAttention(self.embed_dim, self.num_heads)

        # Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)

        self.dropout = nn.Dropout(0.1)

        # Layer norms for intermediate features
        self.intermediate_layer_norms = nn.ModuleDict({
            name: self.backbone.ln1 if name == 'layer31' else nn.LayerNorm(embed_dim)
            for name in self.return_layers.values()
        })

        # Layer normalization
        self.layer_norm = nn.LayerNorm([embed_dim, 24, 18])

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
        intermediate_outputs, model_output = self.extract_layers(x)

        # Apply LayerNorm to the extracted features
        for feature_name in intermediate_outputs.keys():
            intermediate_outputs[feature_name] = self.intermediate_layer_norms[feature_name](intermediate_outputs[feature_name])
            intermediate_outputs[feature_name] = intermediate_outputs[feature_name].view(batch_size, num_frames, self.embed_dim, 24, 18)

        # reshape model output
        #model_output = model_output.reshape(batch_size*num_frames, -1, self.embed_dim)

        # concatenate intermediate outputs and model output
        intermediate_outputs['model_output'] = model_output
        intermediate_outputs['model_output'] = intermediate_outputs['model_output'].view(batch_size, num_frames, self.embed_dim, 24, 18)
                

        # Feature Fusion
        x = self.feature_fusion(intermediate_outputs)
        
        # Reshape to separate frames
        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18) # [batch_size, num_frames, 384, 24, 18]  ß

        # Adaptive Frame Weighting
        x, frame_weights = self.adaptive_weighting(x)

        x = self.dropout(x)
        
        # Cross-Attention
        center_frame_idx = num_frames // 2
        center_frame = x[:, center_frame_idx]
        context_frames = torch.cat([x[:, :center_frame_idx], x[:, center_frame_idx+1:]], dim=1)

        context_frames = context_frames.view(-1, self.embed_dim, 24*18).permute(2, 0, 1)
        context_frames, _ = self.self_attention(context_frames, context_frames, context_frames)
        context_frames = context_frames.permute(1, 2, 0).view(batch_size, num_frames-1, self.embed_dim, 24, 18)

        # Cross-Attention
        attended_features = self.cross_attention(center_frame, context_frames)
        
        # Layer Norm
        attended_features = self.layer_norm(attended_features)

        # residual connection
        attended_features += center_frame

        # Deconvolution layers
        x = self.deconv_layer(attended_features)

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



