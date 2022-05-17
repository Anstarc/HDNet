import os
from tqdm import trange
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch

from .base import BaseDataset


class COCOSegmentation(BaseDataset):
    NUM_CLASS = 171
    # CAT_LIST = list(range(0, 183))
    # unlabeled = [12,26,29,30,45,66,68,69,71,83,91]

    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(COCOSegmentation, self).__init__(
            root, split, mode, transform, target_transform, pad=0, **kwargs)
        # from pycocotools.coco import COCO
        # from pycocotools import mask
        print('train set')
        # ann_file = os.path.join(root, 'cocostuff-10k-v1.1.json')
        self.img_path = os.path.join(root, 'images')
        # for u in self.unlabeled:
        #     if u in self.CAT_LIST:
        #         self.CAT_LIST.remove(u)
        # self.coco = COCO(ann_file)
        # self.coco_mask = mask
        self.images = []

        if self.split == 'train':
            ids_file = os.path.join(root, 'train.txt')
            ann_file = os.path.join(root, 'train.pth')
        elif self.split == 'val':
            ids_file = os.path.join(root, 'val.txt')
            ann_file = os.path.join(root, 'val.pth')
        elif self.split == 'test':
            ids_file = os.path.join(root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        with open(ids_file) as lines:
            assert lines is not None
            for line in lines:
                self.images.append(line[-16:-5])

        if os.path.exists(ann_file):
            self.masks = torch.load(ann_file)
        else:
            raise NotImplementedError
            # ids = list(self.coco.imgs.keys())
            #             print(ids)
            # self.ids = self._preprocess(ids, ann_file)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # coco = self.coco
        img_id = self.images[index]
        # img_metadata = coco.loadImgs(img_id)[0]
        # path = img_metadata['file_name']
        img = Image.open(os.path.join(self.img_path, 'COCO_train2014_0'+img_id+'.jpg')).convert('RGB')
        #         cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        # cocotarget = self.coco.imgToAnns[img_id]
        #         mask = Image.fromarray(self._gen_seg_mask(
        #             cocotarget, img_metadata['height'], img_metadata['width']))
        mask = Image.fromarray(self.masks[int(img_id)])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return torch.from_numpy(target).long()


    def _gen_seg_mask(self, target, h, w, img_id):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        categ = []
        for instance in target:
            #             try:
            #                 if instance['image_id'] == img_id:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            #             print('right instance image_id',instance['image_id'])
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            categ.append(cat)
            if cat in self.CAT_LIST:
                #                 c = self.CAT_LIST.index(cat)
                c = cat
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        #                 else:
        #                     print('wrong instance image_id',instance['image_id'])
        #             except ValueError:
        #                 print('valueError')
        #         decode_segmap(mask, 'coco', True)
        #         color_unique = np.unique(mask)
        #         print(color_unique)
        #         print(np.shape(mask))
        #         print('category', categ)
        #         for i in color_unique:
        #             print(coco_labels()[i])
        return mask

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        masks = {}
        j = 1
        for i in tbar:
            img_id = ids[i]
            #             cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            cocotarget = self.coco.imgToAnns[img_id]
            img_metadata = self.coco.loadImgs(img_id)[0]
            print('image_id', img_id)
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'], img_id)
            masks[img_id] = mask
            new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(masks, ids_file)
        return new_ids
