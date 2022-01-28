""" 
MRLA: Multi-head Recurrent Layer Attention
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .efficientnet_blocks import SqueezeExcite
from .efficientnet_builder import EfficientNetBuilder_CRLA, EfficientNetBuilder_MRLA
from .efficientnet_builder import decode_arch_def, efficientnet_init_weights,\
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from .features import FeatureInfo, FeatureHooks
from .helpers import build_model_with_cfg, default_cfg_for_features
from .layers import create_conv2d, create_classifier
from .registry import register_model

__all__ = ['EfficientNet_MRLA']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    
    'efficientnet_mrla_b0': _cfg(
        input_size=(3, 224, 224)),
    'efficientnet_mrla_b1': _cfg(
        # test_input_size=(3, 256, 256), crop_pct=1.0),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882), 
    'efficientnet_mrla_b2': _cfg(
        input_size=(3, 256, 256), pool_size=(8, 8), test_input_size=(3, 288, 288), crop_pct=1.0),

    # 'tf_efficientnet_mrla_b0': _cfg(
    #     input_size=(3, 224, 224)),
    # 'tf_efficientnet_mrla_b1': _cfg(
    #     input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    # 'tf_efficientnet_mrla_b2': _cfg(
    #     input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    
}   


class EfficientNet_MRLA(nn.Module):
    """ (Generic) EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1

    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False,
                 output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None,
                 se_layer=None, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(EfficientNet_MRLA, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder_MRLA(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features)
        self.act2 = act_layer(inplace=True)
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _create_effnet(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = EfficientNet_MRLA
    kwargs_filter = None
    # if kwargs.pop('features_only', False):
    #     features_only = True
    #     kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'global_pool')
    #     model_cls = EfficientNetFeatures
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    # if features_only:
    #     model.default_cfg = default_cfg_for_features(model.default_cfg)
    return model



def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model



@register_model
def efficientnet_mrla_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_mrla_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_mrla_b1(pretrained=False, **kwargs):
    """ EfficientNet-B1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_mrla_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_mrla_b2(pretrained=False, **kwargs):
    """ EfficientNet-B2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_mrla_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


# @register_model
# def tf_efficientnet_mrla_b0(pretrained=False, **kwargs):
#     """ EfficientNet-B0. Tensorflow compatible variant  """
#     kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         'tf_efficientnet_mrla_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
#     return model


# @register_model
# def tf_efficientnet_mrla_b1(pretrained=False, **kwargs):
#     """ EfficientNet-B1. Tensorflow compatible variant  """
#     kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         'tf_efficientnet_mrla_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
#     return model


# @register_model
# def tf_efficientnet_mrla_b2(pretrained=False, **kwargs):
#     """ EfficientNet-B2. Tensorflow compatible variant  """
#     kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         'tf_efficientnet_mrla_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
#     return model
