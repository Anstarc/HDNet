# ======================================
# File: ccnet.py
# Author: Chunpeng Li
# Email: chunpeng.li@bupt.edu.cn
# ======================================

__all__ = ['CCNet', 'get_ccnet']

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
from ..models import BaseNet
try:
    from mmcv.ops import CrissCrossAttention
except ModuleNotFoundError:
    CrissCrossAttention = None

class CCNet(BaseNet):

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, eval=False,
                 **kwargs):
        super(CCNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.eval = eval

        self.head = RCCAModule(2048, 512, nclass)

    def forward(self, x, target=None, epoch=0):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = [self.head(c4)]

        outputs = []
        for i in range(len(x)):
            outputs.append(upsample(x[i], imsize, **self._up_kwargs))

        if self.eval:
            return outputs[0]

        loss = self.criterion(outputs, target)

        return outputs[0], loss, loss, loss

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

def get_ccnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='./pretrain_models', **kwargs):
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
    print('Network NUM_CLASS:', datasets[dataset.lower()].NUM_CLASS)
    print('#####################')
    model = CCNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model
