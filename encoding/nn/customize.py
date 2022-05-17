##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, CosineEmbeddingLoss, \
    BatchNorm2d
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
from .oim_static import OIMLoss
import numpy as np

torch_ver = torch.__version__[:3]

__all__ = ['GlobalAvgPool2d', 'GramMatrix', 'SegmentationLosses', 'View', 'Sum', 'Mean',
           'Normalize', 'PyramidPooling','SegmentationMultiLosses', 'OIMLosses', 'OHEMLosses', 'ASPPModule']

class GlobalAvgPool2d(Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class GramMatrix(Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

def softmax_crossentropy(input, target, weight, size_average, ignore_index, reduce=True):
    return F.nll_loss(F.log_softmax(input, 1), target, weight,
                      size_average, ignore_index, reduce)

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=255):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


class OIMLosses(OIMLoss):
    def __init__(self, num_features, num_classes, momentum=0.01, weight=None, size_average=True, ingore=-1, para=None, start_epoch=0, bg=None):
        super(OIMLosses, self).__init__(num_features, num_classes, momentum, weight, size_average, ingore, para, start_epoch, bg=bg)

    def forward(self, *inputs):

        *outputs, target, epoch = tuple(inputs)
        pred, dist, attention = tuple(outputs)

        return super(OIMLosses, self).forward(dist, target, pred[0], epoch)


class OhemCrossEntropy2d(Module):
    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight=True):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class OHEMLosses(OhemCrossEntropy2d):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(OHEMLosses, self).__init__(ignore_label=ignore_index, thresh=0.7, min_kept=100000, use_weight=True)
        self.nclass = nclass


    def forward(self, *inputs):

        *outputs, target, epoch = tuple(inputs)
        pred, dist, attention = tuple(outputs)

        loss = []
        for p in pred:
            loss.append(super(OHEMLosses, self).forward(p, target))

        return loss[0]



class SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1, ohem_ratio=0.0, cuda_devices=0):
        if ohem_ratio == 0.0:
            reduce = True
        else:
            size_average = False
            reduce = False
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index, reduce=reduce)
        self.nclass = nclass
        self.oim = None
        self.ohem_ratio = ohem_ratio
        self.eps=1e-7
        self.cuda_devices = cuda_devices


    def forward(self, *inputs):

        if self.cuda_devices == 1:
            *preds, dist, target, epoch = tuple(inputs)
            pred = tuple(preds[0])
        else:
            *outputs, target, epoch = tuple(inputs)
            pred, dist, attention = tuple(outputs)

        loss = []
        for p in pred:
            if self.ohem_ratio == 0.0:
                loss.append(super(SegmentationMultiLosses, self).forward(p, target))
            else:
                l = super(SegmentationMultiLosses, self).forward(p, target)
                mask = self._ohem_mask(l, self.ohem_ratio)
                l = l * mask
                loss.append(l.sum() / (mask.sum() + self.eps))


        if self.oim is not None:
            loss.append(self.oim(dist, target, pred[0], epoch))

        if self.cuda_devices == 1:
            return loss
        else:
            return loss[0]

    def set_oim(self, num_features, num_classes, momentum=0.01,
                 weight=None, size_average=True, ingore=-1, para=None, start_epoch=0, bg=None):
        self.oim = OIMLoss(num_features, num_classes, momentum, weight, size_average, ingore, para, start_epoch, bg=bg)

    def _ohem_mask(self, loss, ohem_ratio):
        with torch.no_grad():
            values, _ = torch.topk(loss.reshape(-1),
                                   int(loss.nelement() * ohem_ratio))
            mask = loss >= values[-1]
        return mask.float()

    def set_cebd(self, margin=0.0, reduction='mean', type=''):
        self.margin = margin
        self.reduction = reduction
        self.type = type
    def CosEmbeddingLoss(self, logit, target_, reduction ='mean', type=''):
        n, c, h, w = logit.size()
        target = target_.unsqueeze(1).float()
        target = torch.nn.functional.upsample_nearest(target, [h, w])

        criterion = CosineEmbeddingLoss(margin=self.margin, reduction=reduction)

        target_h1 = target[:, 0, 1:, :]
        target_h2 = target[:, 0, :-1, :]
        target_h = target_h1 - target_h2
        if 'only_cross' in type:
            target_h = (torch.eq(target_h, 0).float() * 1.0 - 1)
        else:
            target_h = (torch.eq(target_h, 0).float() * 2 - 1)
        logit_h1 = logit[:, :, 1:, :]
        logit_h2 = logit[:, :, :-1, :]
        h_loss = criterion(logit_h1, logit_h2, target_h)

        target_w1 = target[:, 0, :, 1:]
        target_w2 = target[:, 0, :, :-1]
        target_w = target_w1 - target_w2
        if 'only_cross' in type:
            target_w = (torch.eq(target_w, 0).float() * 1.0 - 1)
        else:
            target_w = (torch.eq(target_w, 0).float() * 2 - 1)
        logit_w1 = logit[:, :, :, 1:]
        logit_w2 = logit[:, :, :, :-1]
        w_loss = criterion(logit_w1, logit_w2, target_w)

        return h_loss + w_loss

class View(Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs, attention=False, for_sim=False):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        self.attention = attention
        if attention:
            out_channels = 1
        elif for_sim:
            out_channels = in_channels
        else:
            out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        if attention:
            self.conv0 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                    norm_layer(out_channels),
                                    ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)

        if self.attention:
            feat0 = self.conv0(x)
            return torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        else:
            return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class _ASPPModule(Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPPModule(Module):
    def __init__(self, in_channels, out_channels, norm_layer, output_stride=8, flops=False, dilations=[1, 12, 24, 36], gap=True):
        super(ASPPModule, self).__init__()
        self.gap = gap

        if output_stride == 16 and len(dilations) == 4:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8 and len(dilations) == 4:
            dilations = [1, 12, 24, 36]

        self.aspps = nn.ModuleList([])
        self.aspps.append(_ASPPModule(in_channels, out_channels, 1, padding=0, dilation=1, norm_layer=norm_layer))

        for d in dilations[1:]:
            if d == 0:
                continue
            else:
                self.aspps.append(_ASPPModule(in_channels, out_channels, 3, padding=d, dilation=d, norm_layer=norm_layer))


        # while dilations[1] <= 1:
        #     dilations.append(dilations.pop(0))
        # self.aspp1 = _ASPPModule(in_channels, out_channels, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        # self.aspp2 = _ASPPModule(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        # self.aspp3 = _ASPPModule(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        # self.aspp4 = _ASPPModule(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)
        #
        #
        if gap:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                                                 norm_layer(out_channels),
                                                 nn.ReLU())

        self._init_weight()
        self.flops = flops

    def forward(self, x, cat=True):

        aggregation = []
        for aspp in self.aspps:
            aggregation.append(aspp(x))

        if self.gap:
            g = self.global_avg_pool(x)
            if not self.flops:
                aggregation.append(F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True))

            else:
                aggregation.append(x)

        x = torch.cat(aggregation, dim=1)
        if self.flops:
            if self.gap:
                return x, g
        return x

        # x1 = self.aspp1(x)
        # x2 = self.aspp2(x)
        # x3 = self.aspp3(x)
        # x4 = self.aspp4(x)
        # x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

