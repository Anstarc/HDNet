from .base import *
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pcontext_detail import ContextSegmentation_detail as ContextSegmentation_detail
from .cityscapes import CityscapesSegmentation
from .coco import COCOSegmentation
from .utils import *
from .ecu import ECUSegmentation
from .dark import DarkSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pcontext_detail': ContextSegmentation_detail,
    'cityscapes': CityscapesSegmentation,
    'coco': COCOSegmentation,
    'ecu': ECUSegmentation,
    'dark': DarkSegmentation
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
