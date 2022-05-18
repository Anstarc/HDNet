import torch
from torch.nn.parallel.scatter_gather import gather
import os

def weighted_sum(losses, weights, gradients=True, norm=False):
    if len(weights) == 0:
        for i in range(len(losses)):
            weights.append(1.)
    else:
        assert len(weights) == len(losses), f'len weight:{len(weights)}, len loss:{len(losses)}'
    while len(weights) < len(losses):
        weights.append(1.)

    if gradients:
        loss = losses[0] * weights[0]
        for i in range(1, len(losses)):
            loss += losses[i] * weights[i]
        if norm:
            loss /= sum(weights)

    else:
        with torch.no_grad():
            loss = 0.0
            for i in range(0, len(losses)):
                loss += losses[i].item() * weights[i]
            if norm:
                loss /= sum(weights)
    return loss


def init_loss(length):
    loss = []
    for i in range(length):
        loss.append(0.0)
    return tuple(loss)

def avg_loss(loss, length):
    a_loss = []
    for l in loss:
        a_loss.append(l/length)
    return tuple(a_loss)

def get_cuda_device():
    cuda = os.environ['CUDA_VISIBLE_DEVICES']
    cuda = cuda.split(',')
    devices = []
    for i in cuda:
        devices.append(int(i))
    return devices

def gather_loss(losses):
    losses = losses.mean(dim=0)
    loss = []
    for i in range(losses.size(0)):
        loss.append(losses[i])
    return loss

def print_shape(x, level=''):
    print(f'var {level}')
    if type(x) == list or type(x) == tuple:
        print(type(x), len(x))
        for i in range(len(x)):
            print_shape(x[i], f'{level}.{i}')
    else:
        print(x.size())
        if len(x.size()) == 0:
            print(x)
