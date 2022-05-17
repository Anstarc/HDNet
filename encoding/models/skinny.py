# ======================================
# File: skinny.py
# Author: Chunpeng Li
# Email: chunpeng.li@bupt.edu.cn
# ======================================

from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize

from ..models import BaseNet

from encoding.utils import count_ops
# from torchstat import stat


__all__ = ['Skinny', 'get_skinny']

class Skinny(BaseNet):

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, eval=False,
                 **kwargs):
        super(Skinny, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.evaluation = eval

        self.levels = 6 + 1
        self.initial_filters = 19
        kernel_size = 3

        self.encoder_layers = nn.ModuleList([])
        for i in range(1, self.levels):
            if i == 1:
                in_channels = 3
            else:
                in_channels = get_filters_count(i-1, self.initial_filters)
                in_channels = (in_channels//4)*4
            out_channels = get_filters_count(i, self.initial_filters)
            self.encoder_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                inception_module(out_channels, out_channels//4)
            ))
        self.decoder_layers = nn.ModuleList([])
        self.decoder_inception_layers = nn.ModuleList([])
        self.decoder_dense_layers = nn.ModuleList([])
        for i in range(self.levels-2, 0, -1):
            out_channels = get_filters_count(i, self.initial_filters)
            in_channels = get_filters_count(i+1, self.initial_filters)
            in_channels = (in_channels//4)*4
            self.decoder_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),

            ))
            dense_in_channels = (out_channels//4)*4
            self.decoder_dense_layers.append(dense_block(dense_in_channels, out_channels, kernel_size))

            dense_out_channels = out_channels//2 + out_channels//4 + out_channels//8 + dense_in_channels
            self.decoder_inception_layers.append(inception_module(dense_out_channels+out_channels, out_channels//4))

        self.classifier = nn.Sequential(
            nn.Conv2d((self.initial_filters//4)*4, self.initial_filters, kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.Conv2d(self.initial_filters, self.initial_filters//2, kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.Conv2d(self.initial_filters//2, nclass, kernel_size, padding=kernel_size//2),
        )

    def forward(self, x, target=None, epoch=0):
        imsize = x.size()[2:]

        xx = [x]
        for i in range(0, self.levels-1):
            if i > 0:
                pool = nn.functional.max_pool2d(xx[-1], 2)
                xx.append(self.encoder_layers[i](pool))
            else:
                xx.append(self.encoder_layers[i](xx[0]))
        for i in range(self.levels-2, 0, -1):
            xx[i+1] = upsample(xx[i+1], xx[i].size()[2:], **self._up_kwargs)
            xx[i+1] = self.decoder_layers[self.levels-2-i](xx[i+1])

            xx[i] = self.decoder_dense_layers[self.levels-2-i](xx[i])

            xx[i] = torch.cat([xx[i], xx[i+1]], dim=1)
            xx[i] = self.decoder_inception_layers[self.levels-2-i](xx[i])

        outputs = [self.classifier(xx[1])]
        # for i in range(len(x)):
        #     outputs.append(upsample(x[i], imsize, **self._up_kwargs))
        if self.evaluation:
            return outputs[0]

        # loss = nn.functional.binary_cross_entropy_with_logits(outputs[0], target.unsqueeze(1).float()).reshape(1,1)
        # output = nn.functional.sigmoid(outputs[0])
        # outputs[0] = torch.cat([1-output, output], dim=1)

        loss = self.criterion(outputs, target)
        # if 'oim' in self.try_case.dist_loss:
        #     loss.append(self.oim(attention_rst, target, outputs[0], epoch))

        return outputs[0], outputs[0], outputs[0], loss

class inception_module(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.functional.leaky_relu):
        super(inception_module, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_31 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_32 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_51 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_52 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.max_pool = nn.MaxPool2d(2, stride=1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x2 = self.conv_31(x)
        x2 = self.conv_32(x2)
        x2 = self.activation(x2)
        x3 = self.conv_51(x)
        x3 = self.conv_52(x3)
        x3 = self.activation(x3)
        x4 = self.max_pool(x)
        x4 = upsample(x4, x.size()[2:], mode='bilinear', align_corners=True)
        x4 = self.conv_pool(x4)
        x4 = self.activation(x4)
        return torch.cat([x1,x2,x3,x4], dim=1)

class dense_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.functional.leaky_relu):
        super(dense_block, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.conv2 = nn.Conv2d(in_channels, out_channels//4, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(in_channels, out_channels//8, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm2d(out_channels//8)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.activation(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.activation(x3)
        return torch.cat([x1,x2,x3,x], dim=1)

def get_filters_count(level: int, initial_filters: int) -> int:
    return 2**(level-1)*initial_filters


def get_skinny(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
    model = Skinny(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model
