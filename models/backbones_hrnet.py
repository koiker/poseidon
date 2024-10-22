import torch
import torchvision.models as models
import yaml
import os
from .HRNet.hrnet import get_cls_net

class Backbones:
    def __init__(self, config, device):
        self.backbone_name = config.MODEL.BACKBONE
        self.device = device
        self.freeze_backbone = config.MODEL.FREEZE_BACKBONE
        self.pretrained = config.MODEL.PRETRAINED
        self.backbone = self.get_backbone().to(self.device)
        
        if self.freeze_backbone:
            self.freeze_parameters()
        
        print("Backbone: ", self.backbone_name)

    def get_backbone(self):
        model = None
        
        if self.backbone_name == 'resnet152':
            if self.pretrained != '':
                model = models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V2")
            else:
                model = models.resnet152(weights=None)
        elif self.backbone_name == 'resnet50':
            if self.pretrained != '':
                model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
            else:
                model = models.resnet50(weights=None)
        elif self.backbone_name == 'resnet34':
            if self.pretrained != '':
                model = models.resnet34(weights='DEFAULT')
            else:
                model = models.resnet34(weights=None)
        elif 'hrnet' in self.backbone_name.lower():
            model = self.get_hrnet()
        # Add other backbones as needed
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Remove the classifier for resnets
        if model is not None and 'hrnet' not in self.backbone_name.lower():
            model.fc = torch.nn.Identity()

        return model

    def get_hrnet(self):
        hrnet_config_map = {
            'hrnet-w18': 'cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w18-small-v1': 'cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w18-small-v2': 'cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w30': 'cls_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w32': 'cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w40': 'cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w44': 'cls_hrnet_w44_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w48': 'cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
            'hrnet-w64': 'cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        }

        hrnet_config_pretrained = {
            'hrnet-w18': 'hrnetv2_w18_imagenet_pretrained.pth',
            'hrnet-w18-small-v1': '',
            'hrnet-w18-small-v2': '',
            'hrnet-w30': '',
            'hrnet-w32': '',
            'hrnet-w40': '',
            'hrnet-w44': '',
            'hrnet-w48': '',
            'hrnet-w64': ''
        }

        if self.backbone_name not in hrnet_config_map:
            raise ValueError(f"Unsupported HRNet backbone: {self.backbone_name}")

        config_file = hrnet_config_map[self.backbone_name]
        pretrained_path = hrnet_config_pretrained[self.backbone_name]

        print("current path: s", os.getcwd())


        config_file = os.path.join(os.getcwd(),'models', 'HRNet', 'configs', config_file)
        pretrained_path = os.path.join(os.getcwd(),'models','HRNet', 'pretrained', pretrained_path)

        if not os.path.isfile(config_file):
            raise ValueError(f"HRNet config file not found: {config_file}")
        
        if not os.path.isfile(pretrained_path):
            raise ValueError(f"HRNet pretrained weights not found: {pretrained_path}")
        
        with open(config_file, 'r') as f:
            hrnet_config = yaml.load(f, Loader=yaml.FullLoader)
        
        model = get_cls_net(hrnet_config)
        
        if self.pretrained and os.path.isfile(pretrained_path):
            model.init_weights(pretrained=pretrained_path)
        
        return model

    def freeze_parameters(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_model_info(self):
        num_params = sum(p.numel() for p in self.backbone.parameters())
        output_size = self.backbone(torch.rand(1, 3, 224, 244).to(self.device)).size() # torch.size([1, 1000])
        output_size = output_size[1] 

        return num_params, output_size