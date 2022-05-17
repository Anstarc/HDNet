from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .danet import *
from .hdnet import *
from .emanet import *
from .ccnet import *
from .skinny import *
from .deeplabv3p import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'danet': get_danet,
        'hdnet': get_hdnet,
        'emanet': get_emanet,
        'ccnet': get_ccnet,
        'skinny': get_skinny,
        'deeplabv3p': get_deeplabv3p,
    }
    return models[name.lower()](**kwargs)
