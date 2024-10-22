import torch
import torchvision.models as models
import yaml
import os
from mmpretrain import list_models, get_model
from torchvision.models._utils import IntermediateLayerGetter

class Backbones:
    def __init__(self, config, device):
        self.backbone_name = config.MODEL.BACKBONE
        self.device = device
        self.freeze_backbone = config.MODEL.FREEZE_BACKBONE
        self.pretrained = config.MODEL.PRETRAINED
        self.backbone = self.get_backbone()
        self.backbone = self.backbone.to(device)
        self.return_layers = None

        if self.freeze_backbone:
            self.freeze_parameters()
        
        print("Backbone: ", self.backbone_name)


    def get_backbone(self):
        model = None
        
        if self.backbone_name == 'resnet152':
            self.return_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            if self.pretrained != '':
                model = models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V2")
            else:
                model = models.resnet152(weights=None)
        elif self.backbone_name == 'resnet101':
            self.return_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            if self.pretrained != '':
                model = models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V2")
            else:
                model = models.resnet101(weights=None)
        elif self.backbone_name == 'resnet50':
            self.return_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            if self.pretrained != '':
                model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
            else:
                model = models.resnet50(weights=None)
        elif self.backbone_name == 'resnet34':
            self.return_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            if self.pretrained != '':
                model = models.resnet34(weights='DEFAULT')

            else:
                model = models.resnet34(weights=None)
        elif len(list_models("*" + self.backbone_name + "*" )) > 0: # pretrained model from mmpretrain
            model_dict = {
                    'hrnet-w18': 'hrnet-w18_3rdparty_8xb32_in1k',
                    'hrnet-w30': 'hrnet-w30_3rdparty_8xb30_in1k',
                    'hrnet-w32': 'hrnet-w32_3rdparty_8xb32_in1k',
                    'hrnet-w40': 'hrnet-w40_3rdparty_8xb40_in1k',
                    'hrnet-w44': 'hrnet-w44_3rdparty_8xb44_in1k',
                }
            if self.backbone_name in model_dict:
                model = get_model(model_dict[self.backbone_name], head=None, pretrained=True)
            else:
                print(f"Model {self.backbone_name} not found in the model_dict.")

        # Remove the classifier for resnets
        if model is not None and 'resnet' in self.backbone_name.lower():
            model.fc = torch.nn.Identity()

        return IntermediateLayerGetter(model=model, return_layers={layer: layer for layer in self.return_layers})

    def freeze_parameters(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_model_info(self):
        # num_params = sum(p.numel() for p in self.backbone.parameters())
        # output_size = self.backbone(torch.rand(1, 3, 224, 244).to(self.device)).size() # torch.size([1, 1000])
        # output_size = output_size[1] 

        # return num_params, output_size

        # This method needs to be updated to handle multiple outputs
        num_params = sum(p.numel() for p in self.backbone.parameters())
        dummy_input = torch.rand(1, 3, 224, 224).to(self.device)
        outputs = self.backbone(dummy_input)
        output_sizes = {k: v.shape[1] for k, v in outputs.items()}
        return num_params, output_sizes