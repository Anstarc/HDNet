# ======================================
# File: hdnet.py
# Author: Chunpeng Li
# Email: chunpeng.li@bupt.edu.cn
# ======================================

from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize

from ..nn import Attention
from ..models import BaseNet, SegmentationMultiLosses

from encoding.utils import count_ops
# from torchstat import stat
import kornia


__all__ = ['HDNet', 'get_hdnet']


def round_filter(filters, width_coe, depth_divisor, min_depth):
    if not width_coe:
        return filters
    filters *= width_coe
    min_depth = min_depth or depth_divisor
    new_filter = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filter < 0.9 * filters:  # prevent rounding by more than 10%
        new_filter += depth_divisor
    return int(new_filter)

def get_channels(backbone):
    if 'resnest' in backbone:
        return [2048, 1024, 512, 256, 64, 64]
    elif 'resnet' in backbone:
        return [2048, 1024, 512, 256, 128, 64]
    elif 'efficient' in backbone:
        if 'b6' in backbone:
            wcoe = 1.8
        elif 'b4' in backbone:
            wcoe = 1.4
        elif 'b3' in backbone:
            wcoe = 1.2
        elif 'b2' in backbone:
            wcoe = 1.1
        elif 'b1' in backbone or 'b0' in backbone:
            wcoe = 1.0
        elif 'efficientnet_v2s' in backbone:
            wcoe = 1.0

        settings = [320,112,40,24,16,8]
        if 'efficientnet_v2s' in backbone:
            settings = [272, 160, 64, 48, 24, 24]
        channels = []
        for s in settings:
            channels.append(round_filter(s, wcoe, 8, None))
        return channels
    elif 'mobilenetv3_small' in backbone:
        return [432, 40, 24, 16, 16, 16]
    else:
        return [2048, 1024, 512, 256, 64, 64]


class HDNet(BaseNet):

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, eval=False, skin=False, ft=True,
                 **kwargs):

        super(HDNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.evaluation = eval
        self.aux = aux
        self.skin = skin
        self.ft = ft
        if self.skin:
            self.edge = True

        all_channels = get_channels(backbone)
        channels = all_channels[1:]
        last_channel = channels[0]

        if aux:
            self.head = Attention(all_channels[0], last_channel, norm_layer)
            self.decoder = nn.ModuleList([])
            for i in range(4):
                self.decoder.append(nn.Sequential(
                    nn.Conv2d(last_channel+channels[i], channels[i+1], kernel_size=1),
                    norm_layer(channels[i+1]),
                    nn.ReLU(),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1)
                ))
                last_channel = channels[i+1]
            self.decoder.append(nn.Sequential(
                nn.Conv2d(3 + channels[-1], nclass, kernel_size=1),
                norm_layer(nclass),
                nn.ReLU(),
                nn.Conv2d(nclass, nclass, kernel_size=3, padding=1)
            ))

            self.decoder.append(nn.Sequential(
                nn.Conv2d(3 + channels[-1], 3, kernel_size=1),
                norm_layer(3),
                nn.ReLU(),
                nn.Conv2d(3, 1, kernel_size=3, padding=1)
            ))

        else:
            self.head = Attention(all_channels[0], nclass, norm_layer)

        if self.skin and self.ft:
            self.criterion = SegmentationMultiLosses(nclass=nclass, weight=torch.FloatTensor([1,1.5,1.5,2]))

        self._init_weight()

    def forward(self, x, target=None, epoch=0):
        imsize = x.size()[2:]
        origin = x

        c0, c1, c2, c3, c4 = self.base_forward(x, True)
        x, attention_rst, attention_map = self.head(c4)
        x = list(x)

        if self.aux:
            x.append(self.decoder[0](torch.cat([c3,upsample(x[0], c3.size()[2:], **self._up_kwargs)], dim=1)))
            x.append(self.decoder[1](torch.cat([c2,upsample(x[-1], c2.size()[2:], **self._up_kwargs)], dim=1)))
            x.append(self.decoder[2](torch.cat([c1,upsample(x[-1], c1.size()[2:], **self._up_kwargs)], dim=1)))
            x.append(self.decoder[3](torch.cat([c0,upsample(x[-1], c0.size()[2:], **self._up_kwargs)], dim=1)))
            x.append(self.decoder[4](torch.cat([origin,upsample(x[-1], imsize, **self._up_kwargs)], dim=1)))
            outputs = [x[-1]]
        else:
            outputs = []
            for i in range(len(x)):
                outputs.append(upsample(x[i], imsize, **self._up_kwargs))

        if self.evaluation:
            return outputs[0]

        loss = self.criterion(outputs, target)

        if self.skin:
            edge = self.decoder[5](torch.cat([origin,upsample(x[-2], imsize, **self._up_kwargs)], dim=1))
            sobel = kornia.filters.sobel(target.unsqueeze(1).float())
            sobel = sobel.gt(0).float()
            bce = nn.BCEWithLogitsLoss()
            loss = torch.cat([loss, bce(edge, sobel).reshape(1,1)], dim=1)

        return outputs[0], attention_rst, attention_map, loss

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_hdnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ..datasets import *
    print('#####################')
    print('Network NUM_CLASS:',datasets[dataset.lower()].NUM_CLASS)
    print('#####################')
    model = HDNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model
