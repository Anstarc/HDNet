import torch
from torch import nn
from torch.nn import functional as F, Parameter

from ..nn import CAM_Module


def pyramid_pool(x, pool_size, pool_type):
    pool_rst = []

    for type_ in pool_type:
        if len(pool_size) == 0:
            pool_rst.append(padding_pool(x, 0, type_))
        elif len(pool_size) > 0:
            for size in pool_size:
                pool_rst.append(padding_pool(x, size, type_))

    if len(pool_rst) > 0:
        return pool_rst
    else:
        raise ValueError('len(pool_rst) = ' + str(len(pool_rst)))


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, vis=False):
        super(Attention, self).__init__()
        self.vis = vis
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.sa = LocationAwareAttention(in_channels, norm_layer)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat, attention = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        if self.vis:
            attention.append(sa_conv)
            attention.append(sc_conv)
            attention.append(feat_sum)

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        return tuple(output), feat_sum, attention


class LocationAwareAttention(nn.Module):

    def __init__(self, in_channels, norm_layer, vis=False):
        super(LocationAwareAttention, self).__init__()
        self.vis = vis
        self.norm_layer = norm_layer
        self.pool_size = [3, 5, 7, 9, 11, 15, 19]
        self.pool_type = ['avg', 'max']

        self.define_fuse()
        self.similarity = nn.CosineSimilarity()
        self.bottle_neck = nn.Conv2d(in_channels//4, in_channels//16, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.init_para()

    def spatial_pool(self, x_k, size):
        pool_rst = []

        for avg_pool in self.avg_pools:
            pool_rst.append(F.upsample_nearest(avg_pool(x_k), size))
        for max_pool in self.max_pools:
            pool_rst.append(F.upsample_nearest(max_pool(x_k), size))

        return pool_rst

    def define_fuse(self):
        in_channel = max(len(self.pool_size), 1) * len(self.pool_type)
        out_channel = 1
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            self.norm_layer(out_channel),
            nn.ReLU(inplace=True),
        )

    def fuse_func(self, pools):
        if len(pools) < 1:
            raise RuntimeError
        elif len(pools) == 1:
            return pools[0]
        else:
            return self.fuse(torch.cat(pools, 1))

    def fuse_mm(self, pools, q):
        a = []
        x = []
        pool = []

        for p in pools:
            f = self.similarity(p, q).unsqueeze_(1)
            x.append(f)
            if self.try_case.vis:
                a.append(f.clone().cpu().detach())

            pool.append(p)

        x = self.fuse(torch.cat(x, 1))

        a.append(x.clone().cpu().detach())
        return x, a

    def forward(self, x):
        out = x
        x_value = x
        batch, channel, height, width = x.size()

        x = self.bottle_neck(x)
        x_k = x
        x_q = x

        pools = pyramid_pool(x_k, self.pool_size, self.pool_type)
        context, attention = self.fuse_mm(pools, x_q)

        if self.try_case.vis:
            attention.append(x_value.clone().cpu().detach())

        x_value = x_value.mul(context)
        context = context.view(batch, 1, height, width)

        context_out = self.out_conv(x_value)

        out = out + context_out

        if self.try_case.vis:
            attention.append(context_out.clone().cpu().detach())
            attention.append(out.clone().cpu().detach())

        return out, context


    def init_para(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, self.norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

