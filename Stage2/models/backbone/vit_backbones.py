# Modified from https://github.com/facebookresearch/mae
# https://github.com/gupta-abhay/setr-pytorch
# 参见SETR-PUP

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import os

import timm.models.vision_transformer

class ViTBasePatch16Encoder(nn.Module):
    def __init__(self, img_size=256, output_channels=256, pretrained=None):
        super(ViTBasePatch16Encoder, self).__init__()
        self.vit = vit_base_patch16(img_size=img_size)
        self.pretrained = pretrained
        if self.pretrained is not None:
            self._load_pretrained()
        self.conv_net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.vit.embed_dim, out_channels=self.vit.embed_dim,
                kernel_size=1, stride=1, padding=self._get_padding('VALID', (1, 1),),
                ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # TVDeeplabRes101Encoder.localconv
            nn.Conv2d(self.vit.embed_dim, output_channels, kernel_size=1, stride=1, bias=False)
        )

    def _load_pretrained(self):
        assert os.path.isfile(self.pretrained), "File not found: {}".format(self.pretrained)
        print(f'[_load_pretrained] Attempting to load {self.pretrained}.')
        # 自定义加载方式
        m = torch.load(self.pretrained)
        msd = m['model']
        vsd = self.vit.state_dict()
        # print('vsd.keys() - msd.keys():', set(vsd.keys()) - set(msd.keys()))
        # print('msd.keys() - vsd.keys():', set(msd.keys()) - set(vsd.keys()))
        skip_keys = list(set(vsd.keys()) - set(msd.keys()))
        if len(skip_keys)>0:
            print(f'[Warning] Skipping {len(skip_keys)} keys: {skip_keys}')
        del_keys = []
        for k in list(set(vsd.keys())-set(skip_keys)):
            if msd[k].shape!=vsd[k].shape:
                print(f'{k} does not match:\n    msd[{k}].shape is {msd[k].shape}\n    vsd[{k}].shape is {vsd[k].shape}')
                if k == 'pos_embed':
                    print(f'    resize {k}: {msd[k].shape} -> {vsd[k].shape}')
                    msd[k] = timm.models.vision_transformer.resize_pos_embed(msd[k], self.vit.pos_embed, self.vit.num_tokens, self.vit.patch_embed.grid_size)
                else:
                    del_keys.append(k)
        if len(del_keys)>0:
            print(f'[Warning] Deleting {len(del_keys)} keys: {del_keys}')
        for key in del_keys:  # 遍历要删除字段的list
            del msd[key]  # 删除预训练权重的key和对应的value
        missing_keys, unexpected_keys = self.vit.load_state_dict(msd, strict=False)
        if len(missing_keys)>0:
            print(f'[Warning] Missing {len(missing_keys)} keys: {missing_keys}')
        if len(unexpected_keys)>0:
            print(f'[Warning] Unexpected {len(unexpected_keys)} keys: {unexpected_keys}')
        print(f'[_load_pretrained] {self.pretrained} loaded.')

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.vit.patch_embed.img_size[0] / self.vit.patch_embed.patch_size[0]),
            int(self.vit.patch_embed.img_size[1] / self.vit.patch_embed.patch_size[1]),
            self.vit.embed_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, low_level=False):
        if low_level:
            raise NotImplementedError('Low level encoder is not implemented yet.')
        fts = self.vit.forward_features(x)
        fts = self._reshape_output(fts)
        high_level_fts = self.conv_net(fts)
        return high_level_fts


class VisionTransformerBackbone(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer
    """
    def __init__(self, **kwargs):
        super(VisionTransformerBackbone, self).__init__(**kwargs)
        # 去除ViT的head层, 因为作为backbone提特征就行
        # v.head=nn.Sequential()
        del self.head

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 1:, :] # 不返回cls_token


def vit_base_patch16(**kwargs):
    model = VisionTransformerBackbone(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformerBackbone(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformerBackbone(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model