# recurrent layer attention, groupwise
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.la_module import mla_layer
from .modules.eca_module import eca_layer
from .modules.se_module import se_layer
# from .utils.drop import DropPath

from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES

from mmcv.runner import BaseModule


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class mrla_module(BaseModule):
    dim_pergroup = 32
    
    def __init__(self, input_dim):
        super(mrla_module, self).__init__()
        self.la = mla_layer(input_dim=input_dim, dim_pergroup=self.dim_pergroup)
        self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  # nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, xt, ot_1):
        atten_t = self.la(xt)
        out = atten_t + self.lambda_t.expand_as(ot_1) * ot_1 # o_t = atten(x_t) + lambda_t * o_{t-1}
        return out

#=========================== define bottleneck ============================
# class LABottleneck(nn.Module):
class LABottleneck(BaseModule):
    expansion = 4

    def __init__(self, inplanes, planes, 
                 stride=1, downsample=None, 
                 SE=False, ECA_size=None, 
                 groups=1, base_width=64, dilation=1, 
                 norm_layer=nn.BatchNorm2d, drop_path=0.0, 
                 init_cfg=None):
        super(LABottleneck, self).__init__(init_cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
            
        width = int(planes * (base_width / 64.)) * groups
        
        # self.bn0 = norm_layer(inplanes)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        
        # channel attention modules
        self.se = None
        if SE:
            self.se = se_layer(planes * self.expansion, reduction=16)
        self.eca = None
        if ECA_size != None:
            # self.eca = eca_layer(planes * 4, k_size)
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))
            
        # recurrent layer attention module
        self.mrla = mrla_module(input_dim=planes * self.expansion)
        self.bn_mrla = norm_layer(planes * self.expansion)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # nn.Identity() is placeholder

    def forward(self, x):
        identity = x
        
        # x = self.bn0(x)
        # res block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # channel attention
        if self.se != None:
            out = self.se(out)
        if self.eca != None:
            out = self.eca(out) 
        # downsampling for short cut    
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity # x_t = res_t + o_{t-1}
        out = self.relu(out)
        # layer attention
        out = out + self.bn_mrla(self.mrla(out, identity))
        
        return out
    
    
#=========================== define network ============================
@BACKBONES.register_module()
# class ResNet_LA(nn.Module):
class ResNet_LA(BaseModule):
    '''
    frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
        -1 means not freezing any parameters.
    norm_eval (bool): Whether to set norm layers to eval mode, namely,
        freeze running stats (mean and var). Note: Effect on Batch Norm
        and its variants only.
    zero_init_last_bn (bool): Whether to use zero init for last norm layer
        in resblocks to let them behave as identity.
    '''
    def __init__(self, 
                block=LABottleneck, 
                layers=[3, 4, 6, 3], 
                # num_classes=1000, 
                # la_channels=[256, 512, 1024, 2048], 
                SE=False, 
                ECA=None, 
                frozen_stages=-1,
                norm_eval=True,
                style='pytorch',
                zero_init_last_bn=True,  #zero_init_residual=False,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                norm_layer=nn.BatchNorm2d, 
                drop_rate=0.0, 
                drop_path=0.0,
                pretrained=None,
                init_cfg=None
                ):
        super(ResNet_LA, self).__init__(init_cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
        self._norm_layer = norm_layer
        # self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.drop_path = drop_path
        self.zero_init_last_bn = zero_init_last_bn
        
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]

                if self.zero_init_last_bn:
                    # if block is BasicBlock:
                    #     block_init_cfg = dict(
                    #         type='Constant',
                    #         val=0,
                    #         override=dict(name='norm2'))
                    # elif block is Bottleneck:
                    if block is LABottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='bn3'))
        else:
            raise TypeError('pretrained must be a str or None')
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if ECA is None:
            ECA = [None] * 4
        elif len(ECA) != 4:
            raise ValueError("argument ECA should be a 4-element tuple, got {}".format(ECA))
    
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # stages = [None] * 4
        self.layer1 = self._make_layer(block, 64, layers[0], SE=SE, ECA_size=ECA[0], init_cfg=block_init_cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], SE=SE, ECA_size=ECA[1], init_cfg=block_init_cfg, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], SE=SE, ECA_size=ECA[2], init_cfg=block_init_cfg, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], SE=SE, ECA_size=ECA[3], init_cfg=block_init_cfg, stride=2, dilate=replace_stride_with_dilation[2])
        # self.stages = nn.ModuleList(stages)
        
        # classifier head
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # initialization
        ''' 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # elif isinstance(m, (nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_last_bn:
        # if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LABottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)
        '''

    def _make_layer(self, block, planes, blocks, SE, ECA_size, init_cfg, 
                    stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ) # downsampling and change channels for x(identity)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            SE=SE, ECA_size=ECA_size, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, drop_path=self.drop_path, 
                            init_cfg=init_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, drop_path=self.drop_path, 
                                init_cfg=init_cfg))
            
        return nn.Sequential(*layers)
        # return nn.ModuleList(layers)

    # def _forward_impl(self, x):
    def forward_features(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        # for layers in self.stages:
        #     for layer in layers:
        #         x = layer(x)

        # classifier head
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # return x
        return tuple(outs)
    
    def forward(self, x):
        return self.forward_features(x)
        # x = self.forward_features(x)
        
        # # classifier head
        # # x = self.global_pool(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # if self.drop_rate:
        #     x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        # x = self.fc(x)
        # return x
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # if self.deep_stem:
            #     self.stem.eval()
            #     for param in self.stem.parameters():
            #         param.requires_grad = False
            # else:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
                

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 kaiming_init(m)
    #             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
    #                 constant_init(m, 1)

    #         # if self.dcn is not None:
    #         #     for m in self.modules():
    #         #         if isinstance(m, Bottleneck) and hasattr(
    #         #                 m.conv2, 'conv_offset'):
    #         #             constant_init(m.conv2.conv_offset, 0)

    #         if self.zero_init_last_bn:
    #             for m in self.modules():
    #                 if isinstance(m, LABottleneck):
    #                     constant_init(m.bn3, 0)
    #                 # elif isinstance(m, BasicBlock):
    #                 #     constant_init(m.norm2, 0)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet_LA, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


