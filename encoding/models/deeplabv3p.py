# ======================================
# File: deeplabv3p.py
# Author: Chunpeng Li
# Email: chunpeng.li@bupt.edu.cn
# ======================================
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize

from ..nn import ASPPModule


from ..models import BaseNet

__all__ = ['DeepLabV3p', 'get_deeplabv3p']

class DeepLabV3p(BaseNet):
    def get_parameters(self):
        return [self.aspp.parameters(), self.low_level_conv.parameters(), self.classifier.parameters()]

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, eval=False,
                 **kwargs):
        super(DeepLabV3p, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.evaluation = eval

        self.aspp = nn.Sequential(
            ASPPModule(2048, 256, norm_layer=norm_layer),
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, kernel_size=1, stride=1)
        )

    def forward(self, x, target=None, epoch=0):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        c4 = self.aspp(c4)
        c4 = upsample(c4, c1.size()[2:], **self._up_kwargs)
        c1 = self.low_level_conv(c1)
        c4 = torch.cat([c4, c1], dim=1)
        c4 = self.classifier(c4)


        outputs = []
        x = [c4]
        for i in range(len(x)):
            outputs.append(upsample(x[i], imsize, **self._up_kwargs))
        if self.evaluation:
            return outputs[0]

        loss = self.criterion(outputs, target)

        return outputs[0], outputs[0], outputs[0], loss



def get_deeplabv3p(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
    model = DeepLabV3p(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model
