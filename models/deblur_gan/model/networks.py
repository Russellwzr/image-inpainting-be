import torch.nn as nn
import functools
from models.deblur_gan.model.fpn_inception import FPNInception


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_generator(model_config):
    model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config.norm_layer))
    return nn.DataParallel(model_g)



