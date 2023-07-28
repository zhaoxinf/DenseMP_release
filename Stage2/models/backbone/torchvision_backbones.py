"""
Backbones supported by torchvison.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.segmentation.segmentation import _segm_model

class TVDeeplabRes101Encoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """
    def __init__(self, use_coco_init, aux_dim_keep = 64, use_aspp = False):
        super().__init__()
        _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=use_coco_init, progress=True, num_classes=21, aux_loss=None)
        if use_coco_init:
            print("###### NETWORK: Using ms-coco initialization ######")
        else:
            print("###### NETWORK: Training from scratch ######")

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256,kernel_size = 1, stride = 1, bias = False) # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256,kernel_size = 1, bias = False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256] )
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts['out']
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts

class TVDeeplabRes101DenseCLEncoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """
    def __init__(self, pretrain=None, aux_dim_keep = 64, use_aspp = False):
        super().__init__()
        arch_type = 'deeplabv3'
        backbone_name = 'resnet101'
        num_classes = 21
        aux_loss = True
        pretrained_backbone = False
        _model = _segm_model(arch_type, backbone_name, num_classes, aux_loss, pretrained_backbone)
        if pretrain is not None:
            pretrained_backbone_weights = torch.load(pretrain)['state_dict']
            _model.backbone.load_state_dict(pretrained_backbone_weights)
            print("###### NETWORK: DenseCL ResNet101 weights loaded. ######")
            print(f"###### NETWORK: {pretrain} ######")

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256,kernel_size = 1, stride = 1, bias = False) # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256,kernel_size = 1, bias = False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256] )
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts['out']
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts

class TVDeeplabRes101DenseCLEncoderPlus(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """
    def __init__(self, aux_dim_keep = 64, use_aspp = False):
        super().__init__()
        _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
        print("###### NETWORK: Using ms-coco initialization of DeepLabV3 Head and Neck ######")
        densecl_imagenet_resnet101 = 'pretrained_model/densecl/imagenet/densecl_r101_imagenet_200ep.pth'
        pretrained_backbone_weights = torch.load(densecl_imagenet_resnet101)['state_dict']
        _model.backbone.load_state_dict(pretrained_backbone_weights)
        print("###### NETWORK: DenseCL ImageNet ResNet101 weights loaded. ######")

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256,kernel_size = 1, stride = 1, bias = False) # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256,kernel_size = 1, bias = False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256] )
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts['out']
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts

# initial_weights = model.backbone.state_dict()
# for k in initial_weights.keys():
#     if k not in pretrained_weights.keys():
#         print(f'key {k} in pretrained weights do not exists, use initial weights instead.')
#         pretrained_weights[k] = initial_weights[k]
