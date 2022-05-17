from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# import matplotlib.pyplot as plt

class OIM(autograd.Function):
    @staticmethod
    def init(ignore, momentum=0.5):
        OIM.momentum = momentum
        OIM.ignore = ignore

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, inputs, targets, preds, lut):
        ctx.save_for_backward(inputs, targets, preds, lut)

        batch, channel, height, width = inputs.size()
        outputs = inputs.permute(0, 2, 3, 1).reshape(-1, channel)
        outputs = outputs.mm(lut.t())
        outputs = outputs.reshape(batch, height, width, -1).permute(0, 3, 1, 2)

        return outputs

    @staticmethod
    def update(feature, target, pred, lut):
        batch_size, channel, height, width = feature.size()

        current_classes = torch.unique(target)
        current_classes = current_classes[current_classes != OIM.ignore]

        for i in current_classes:
            mask = target.eq(i).float()
            feature_i = feature.mul(mask.unsqueeze(1))

            feature_i = feature_i.permute(0, 2, 3, 1).reshape(-1, channel)
            feature_i = feature_i.sum(dim=0) / mask.sum()

            lut[i] = OIM.momentum * lut[i] + (1. - OIM.momentum) * feature_i
            lut[i] /= lut[i].norm()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets, preds, lut = ctx.saved_tensors
        # grad_inputs = None
        # if self.needs_input_grad[0]:
        batch, channel, height, width = grad_outputs.size()
        outputs = grad_outputs.permute(0, 2, 3, 1).reshape(-1, channel)
        grad_inputs = outputs.mm(lut)
        grad_inputs = grad_inputs.reshape(batch, height, width, -1).permute(0, 3, 1, 2)

        OIM.update(inputs, targets, preds, lut)

        return grad_inputs, None, None, None


# def oim_proc(inputs, targets, preds, lut, ignore, momentum=0.5):
#     OIM.init(lut, ignore, momentum=momentum)
#     return OIM.apply(inputs, targets, preds)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, momentum=0.5,
                 weight=None, size_average=True, ingore=None, para=None, start_epoch=0, scalar=1.0, bg=None):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.ignore = ingore
        self.para = para
        self.start_epoch = start_epoch
        self.bg = bg

        if para is not None:
            self.use_para = True
            if 'gamma' in para.keys():
                self.momentum = para['gamma']

        # self.register_buffer('lut', torch.zeros(num_classes, num_features, requires_grad=False))    # [NCLASS, C]
        # self.lut = torch.zeros(num_classes, num_features, requires_grad=False)

    def resize(self, inputs, targets, preds=None):
        if self.use_para and 'up' in self.para.keys():
            t_size = targets.size()[1:]
            if self.para['up'] == 'bilinear':
                feature = F.upsample(inputs, t_size, mode='bilinear', align_corners=True)
            else:
                feature = F.upsample(inputs, t_size)

            target = targets.clone()
            out_size = feature.size()

            if preds is not None:
                pred = preds.clone()
            else:
                pred = None

        else:
            out_size = inputs.size()
            batch, channel, height, width = out_size
            target = targets.float().unsqueeze(1)
            target = F.upsample(target, out_size[2:]).view(batch, height, width).long()
            feature = inputs.clone()

            if preds is not None:
                pred = F.adaptive_avg_pool2d(preds, out_size[2:])
            else:
                pred = None

        return feature, target, pred, out_size

    def forward(self, lut, inputs, targets, preds=None, epoch=None):

        if epoch < self.start_epoch:
            return torch.tensor(0)

        feature, target, pred, size = self.resize(inputs, targets, preds)

        if self.bg is not None:
            target[target == self.bg] = self.ignore
            # assert self.bg not in torch.unique(target)

        if self.use_para and 'dynimic_gamma' in self.para.keys():
            momentum = self.get_gamma(epoch)
        else:
            momentum = self.momentum

        OIM.init(self.ignore, momentum=momentum)
        outputs = OIM.apply(feature, target, pred, lut)
        # outputs = oim_proc(feature, target, pred, self.lut, self.ignore, momentum=momentum)

        loss = F.cross_entropy(outputs, target, weight=self.weight,
                               size_average=self.size_average, ignore_index=self.ignore)
        return loss

    def get_gamma(self, epoch):
        assert 'total_epoch' in self.para.keys() and epoch is not None
        return self.momentum * pow(
            (1 - 1.0 * (epoch - self.start_epoch) / (self.para['total_epoch'] - self.start_epoch)), 0.9)


# test OIM
if __name__ == "__main__":
    # from encoding.parallel import DataParallelModel
    import torch.optim as optim
    import os


    def get_cuda_device():
        cuda = os.environ['CUDA_VISIBLE_DEVICES']
        cuda = cuda.split(',')
        devices = []
        for i in cuda:
            devices.append(int(i))
        return devices


    class Net(nn.Module):
        def __init__(self, input_channel, hd_channel, nclass):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(input_channel, hd_channel, 1)
            self.conv2 = nn.Conv2d(hd_channel, nclass, 1)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            loss1 = nn.functional.cross_entropy(x2, y)
            return x2, x1, loss1


    s_size = 2
    batch_size = 2
    input_channel = 3
    nclass = 5
    hd_channel = 4
    cuda = True

    x = torch.rand(batch_size, input_channel, s_size, s_size)
    y = torch.randint(0, nclass, [batch_size, s_size, s_size]).long()

    net = Net(input_channel, hd_channel, nclass)
    lut = torch.zeros(nclass, hd_channel, requires_grad=False)
    oimloss = OIMLoss(hd_channel, nclass, para={'dynimic_gamma': '', 'gamma': 0.001, 'total_epoch': 80}, bg=-1,
                      ingore=-1)

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()

    cuda_device = get_cuda_device()
    print('cuda_device', cuda_device)
    # net = DataParallelModel(net, device_ids=cuda_device)
    if cuda:
        net = net.cuda()
        x = x.cuda()
        y = y.cuda()

    for i in range(2):
        output = net(x)
        loss2 = oimloss(lut, output[1], y, output[0], i)
        loss = loss2 + output[2]

        loss.backward()
        optimizer.step()
        print(lut)