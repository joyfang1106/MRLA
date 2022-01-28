""" EfficientNet, MobileNetV3, etc Blocks

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import create_conv2d, drop_path, make_divisible
from .layers.activations import sigmoid

from math import sqrt
from math import log
from .layers.drop import DropPath

__all__ = [
    'SqueezeExcite', 'ConvBnAct', 'DepthwiseSeparableConv', 'InvertedResidual', 'CondConvResidual', 'EdgeResidual',
    'InvertedResidual_CRLA', 'InvertedResidual_MRLA']


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        se_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_fn (Callable): attention gate function
        block_in_chs (int): input channels of containing block (for calculating reduction from)
        reduce_from_block (bool): calculate reduction from block input channels if True
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        divisor (int): make reduction channels divisible by this
    """

    def __init__(
            self, in_chs, se_ratio=0.25, act_layer=nn.ReLU, gate_fn=sigmoid,
            block_in_chs=None, reduce_from_block=True, force_act_layer=None, divisor=1):
        super(SqueezeExcite, self).__init__()
        reduced_chs = (block_in_chs or in_chs) if reduce_from_block else in_chs
        reduced_chs = make_divisible(reduced_chs * se_ratio, divisor)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """
    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, dilation=1, pad_type='',
            skip=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_path_rate=0.):
        super(ConvBnAct, self).__init__()
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        self.drop_path_rate = drop_path_rate
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, se_ratio=se_ratio, act_layer=act_layer) if has_se else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(
            mid_chs, se_ratio=se_ratio, act_layer=act_layer, block_in_chs=in_chs) if has_se else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x
    
    
###################### add ######################
class mla_layer(nn.Module):
    """
    groupwise layer attention module
    when groups = channels, channelwise (Q(K)' is then pointwise(channelwise) multiplication)
    
    Args:
        input_dim: input channel c (output channel is the same)
        k_size: channel dimension of Q, K
        input : [b, c, h, w]
        output: [b, c, h, w]
        
        Wq, Wk: conv1d
        Wv: conv2d
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
    """
    def __init__(self, input_dim, groups=8, dim_pergroup=None, k_size=None):
        super(mla_layer, self).__init__()
        self.input_dim = input_dim
        self.k_size = k_size
        if (groups == None) and (dim_pergroup == None):
            raise ValueError("arguments groups and dim_pergroup cannot be None at the same time !")
        elif dim_pergroup != None:
            groups = int(input_dim / dim_pergroup)
        else:
            groups = groups
        self.groups = groups
        
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        self._norm_fact = 1 / sqrt(input_dim / groups)
        # nn.ReLU(inplace=True)
        # self.depthwise1 = nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c] 
        K = self.Wk(y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        # Q = Q.chunk(self.groups, dim = -1) # a tuple of g * [b, 1, c/g]
        # K = K.chunk(self.groups, dim = -1)
        # V = V.chunk(self.groups, dim = 1) # a tuple of g * [b, c/g, h, w]
        Q = Q.view(b, self.groups, 1, int(c/self.groups)) # [b, g, 1, c/g]
        K = K.view(b, self.groups, 1, int(c/self.groups)) # [b, g, 1, c/g]
        V = V.view(b, self.groups, int(c/self.groups), h, w) # [b, g, c/g, h, w]
        # Q.is_contiguous()
        
        atten = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        # atten.size() # [b, g, 1, 1]
    
        atten = self.sigmoid(atten.view(b, self.groups, 1, 1, 1)) # [b, g, 1, 1, 1]
        output = V * atten.expand_as(V) # [b, g, c/g, h, w]
        output = output.view(b, c, h, w)
        
        return output  
    
class cla_layer(nn.Module):
    """
    channelwise layer attention module
    change dot product of Q(K)' to the pointwise(channelwise) multiplication
    Args:
        input_dim: input channel c (output channel is the same)
        k_size: channel dimension of Q, K
        input : [b, c, h, w]
        output: [b, c, h, w]
        
        Wq, Wk: conv1d
        Wv: conv2d
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
    """
    def __init__(self, input_dim, k_size=None):
        super(cla_layer, self).__init__()
        self.input_dim = input_dim
        self.k_size = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        # self._norm_fact = 1 / sqrt(input_dim)
        # nn.ReLU(inplace=True)
        # self.depthwise1 = nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c]
        K = self.Wk(y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        
        # atten = (torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact # Q * K.T() [b, 1, 1]
        atten = Q * K # [b, 1, c]
        atten = self.sigmoid(atten.view(b, c, 1, 1)) # [b, c, 1, 1]
        output = V * atten.expand_as(V)
        
        return output    
        
    
class crla_module(nn.Module):
    
    def __init__(self, input_dim):
        super(crla_module, self).__init__()
        self.cla = cla_layer(input_dim=input_dim)
        self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  # nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, xt, ot_1):
        atten_t = self.cla(xt)
        out = atten_t + self.lambda_t.expand_as(ot_1) * ot_1 # o_t = atten(x_t) + lambda_t * o_{t-1}
        return out  
    
class mrla_module(nn.Module):
    
    def __init__(self, input_dim, g_gla=8):
        super(mrla_module, self).__init__()
        self.mla = mla_layer(input_dim=input_dim, groups=g_gla)
        self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  # nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, xt, ot_1):
        atten_t = self.mla(xt)
        out = atten_t + self.lambda_t.expand_as(ot_1) * ot_1 # o_t = atten(x_t) + lambda_t * o_{t-1}
        return out 
    
    
    
class InvertedResidual_CRLA(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0., drop_path_rla=0.2):
        super(InvertedResidual_CRLA, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(
            mid_chs, se_ratio=se_ratio, act_layer=act_layer, block_in_chs=in_chs) if has_se else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs)
        
        if self.has_residual:
            # recurrent layer attention module
            self.crla = crla_module(input_dim=out_chs)
            self.bn_crla = norm_layer(out_chs)
            # self.drop_path_rla = DropPath(drop_path_rla) if drop_path_rla > 0. else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
            
            # layer attention
            o = self.bn_crla(self.crla(x, shortcut))
            x = x + drop_path(o, self.drop_path_rate, self.training)
    
        return x
    
    
class InvertedResidual_MRLA(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0., drop_path_rla=0.2):
        super(InvertedResidual_MRLA, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(
            mid_chs, se_ratio=se_ratio, act_layer=act_layer, block_in_chs=in_chs) if has_se else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs)
        
        if self.has_residual:
            # recurrent layer attention module
            self.mrla = mrla_module(input_dim=out_chs)
            self.bn_mrla = norm_layer(out_chs)
            # self.drop_path_rla = DropPath(drop_path_rla) if drop_path_rla > 0. else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
            
            # layer attention
            o = self.bn_mrla(self.mrla(x, shortcut))
            x = x + drop_path(o, self.drop_path_rate, self.training)
    
        return x

###################### add ######################

class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, num_experts=0, drop_path_rate=0.):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)

        super(CondConvResidual, self).__init__(
            in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, pad_type=pad_type,
            act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size, se_ratio=se_ratio, se_layer=se_layer,
            norm_layer=norm_layer, conv_kwargs=conv_kwargs, drop_path_rate=drop_path_rate)

        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x

        # CondConv routing
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))

        # Point-wise expansion
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(
            self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, pad_type='',
            force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.):
        super(EdgeResidual, self).__init__()
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(
            mid_chs, se_ratio=se_ratio, act_layer=act_layer, block_in_chs=in_chs) if has_se else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == 'expansion':  # after SE, before PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x
