# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import math
import logging
from collections import OrderedDict
from copy import deepcopy

# from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.vision_transformer import default_cfgs, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers.helpers import to_2tuple

# from timm.models.layers import PatchEmbed, DropPath, lecun_normal_
from timm.models.layers import DropPath
from weight_init import lecun_normal_
from math import sqrt
from math import log

# from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from helpers import named_apply


'''
mrla-light version
'''


__all__ = [
    'deit_mrlal_tiny_patch16_224', 'deit_mrlal_small_patch16_224', 'deit_mrlal_base_patch16_224',
]



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
        

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class mrlal_layer(nn.Module):
    """
    when heads = channels, channelwise (Q(K)' is then pointwise(channelwise) multiplication)
    
    Args:
        input_dim: input channel c (output channel is the same)
        k_size: kernel size of conv1d
        input : [b, c, h, w]
        output: [b, c, h, w]
        
        Wq, Wk: conv1d
        Wv: conv2d
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
    """
    def __init__(self, input_dim, heads=None, dim_perhead=None, k_size=None):
        super(mrlal_layer, self).__init__()
        self.input_dim = input_dim
        self.k_size = k_size
        if (heads == None) and (dim_perhead == None):
            raise ValueError("arguments heads and dim_perhead cannot be None at the same time !")
        elif dim_perhead != None:
            heads = int(input_dim / dim_perhead)
        else:
            heads = heads
        self.heads = heads
        
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        self.act_v = nn.GELU()
        self._norm_fact = 1 / sqrt(input_dim / heads)
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
        V = self.act_v(V)
        Q = Q.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        K = K.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        V = V.view(b, self.heads, int(c/self.heads), h, w) # [b, g, c/g, h, w]
        # Q.is_contiguous()
        
        atten = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        # atten.size() # [b, g, 1, 1]
    
        atten = self.sigmoid(atten.view(b, self.heads, 1, 1, 1)) # [b, g, 1, 1, 1]
        output = V * atten.expand_as(V) # [b, g, c/g, h, w]
        output = output.view(b, c, h, w)
        
        return output

    
class mrlal_module(nn.Module):
    
    def __init__(self, input_dim, dim_perhead, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(mrlal_module, self).__init__()
        self.dim_perhead = dim_perhead
        self.mrla = mrlal_layer(input_dim=input_dim, dim_perhead=self.dim_perhead)
        self.lambda_t = nn.Parameter(torch.randn(input_dim))  # nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.normx = norm_layer(input_dim)
        self.normo = norm_layer(input_dim)
        
    def forward(self, xt, ot_1):
        xt = self.normx(xt)
        ot_1 = self.normo(ot_1)
        
        b, n, k = xt.size()
        cls_token, tokens = torch.split(xt, [1, n - 1], dim=1)
        # h, w = int(math.sqrt(n - 1)), int(math.sqrt(n - 1))
        xt = tokens.reshape(b, int(sqrt(n - 1)), int(sqrt(n - 1)), k).permute(0, 3, 1, 2)
        
        xt = self.mrla(xt)
        tokens = xt.flatten(2).permute(0, 2, 1)
        _, ot_1 = torch.split(ot_1, [1, n - 1], dim=1)
        tokens = tokens + self.lambda_t.expand_as(ot_1) * ot_1 # o_t = attn(x_t) + lambda_t * o_{t-1}
        out = torch.cat((cls_token, tokens), dim=1)
        
        return out


class Block(nn.Module):

    def __init__(self, dim, num_heads, dim_mrla, 
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.mrla = mrlal_module(input_dim=dim, dim_perhead=dim_mrla)

    def forward(self, x):
        ot = x
        
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # layer attention
        x = x + self.mrla(x, ot)
        return x


class ViT_mrlal(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12,
                 num_heads=12, dim_mrla=16, 
                 mlp_ratio=4., qkv_bias=True, 
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=nn.GELU, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, dim_mrla=dim_mrla, 
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        # self.avgpool = nn.AdaptiveAvgPool2d((1))
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    # @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=''):
    #     _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        
        # feas = self.avgpool(x[:, 1:, ])
        # feas = torch.flatten(feas, 1)
        # out = x[:, 0] + feas
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    
    
def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
        
        



@register_model
def deit_mrlal_tiny_patch16_224(pretrained=False, **kwargs):
    model = ViT_mrlal(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, dim_mrla=16, 
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_mrlal_small_patch16_224(pretrained=False, **kwargs):
    model = ViT_mrlal(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, dim_mrla=16, 
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_mrlal_base_patch16_224(pretrained=False, **kwargs):
    model = ViT_mrlal(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, dim_mrla=16, 
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


    
if __name__ == '__main__':
    from params_flops import compute_params
    compute_params(deit_mrlal_tiny_patch16_224())