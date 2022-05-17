import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import torchvision

import torch
from .base import BaseDataset

class ECUSegmentation(BaseDataset):
    NUM_CLASS = 2
    BASE_DIR = 'Skinny'
    def __init__(self, root, split='train', mode=None, transform=None, 
                 target_transform=None, **kwargs):
        super(ECUSegmentation, self).__init__(root, split, mode, transform,
                                               target_transform, pad=0, **kwargs)
        self.root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(self.root, 'labels')
        _image_dir = os.path.join(self.root, 'features')

        if mode == 'testval':
            self.split = 'test'

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'train.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'val.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        self.val_names = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform( img, target)
            img = torchvision.transforms.functional.adjust_gamma(img, random.random()+0.5)
        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
            target = self._mask_transform(target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            target = self.target_transform(target)
        return img, target

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')[:,:,0]
        target[target == 255] = 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)
