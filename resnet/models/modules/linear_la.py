"""Implement unmasked linear attention."""

import torch
from torch import nn
from torch.nn import Module
from math import sqrt
from math import log

from .feature_map import elu_feature_map


class LinearLayerAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the queries, keys and values as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, input_dim, feature_map=None, eps=1e-6, k_size=None, svd=False):
        super(LinearLayerAttention, self).__init__()
        query_dimensions = input_dim
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.svd = svd
        # self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        
    #     self.reset_params()
        
    # def reset_params(self):
    #     nn.init.kaiming_normal_(self.Wv, mode='fan_out', nonlinearity='relu')

    def forward(self, x, s, z):
        """
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
        
        s <-- s + feature_map(K)V'
        z <-- z + feature_map(K)
        
        out <-- (feature_map(Q)s) / (feature_map(Q)z)
        """
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c] 
        K = self.Wk(y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        V = V.view(b, 1, c*h*w) # V: [b, 1, chw]
        
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map()
        Q = self.feature_map.forward_queries(Q)
        K = self.feature_map.forward_keys(K)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        # let d=chw, t=1, c=c
        KV = torch.einsum('btc, btd -> bcd', K, V) # [b, c, chw]
        if self.svd:
            s_u, s_s, s_v = s[0], s[1], s[2]
            s = torch.mm(torch.mm(s_u, torch.diag(s_s)), s_v.t())
        s = s + KV

        # Compute the normalizer
        z = z + K # [b, 1, c]
        QZ = 1.0 / torch.einsum('btc, btc -> bt', Q, z+self.eps) # [b, 1]
        # Finally compute and return the new values
        out = torch.einsum("btc, bcd, bt -> btd", Q, s, QZ) # [b, 1, chw]
        out = out.contiguous().view(b, c, h, w)
        
        if self.svd:
            # torch.linalg.svd(A, full_matrices=True, *, out=None)
            # s_u, s_s, s_v = torch.svd(s, full_matrices=False)
            # torch.svd(input, some=True, compute_uv=True, *, out=None)
            s_u, s_s, s_v = torch.svd(s, some=True)
            s = (s_u, s_s, s_v)

        return out, s, z



class linear_cla(Module):
    '''
    Linear Channelwise Layer Attention
    '''
    def __init__(self, input_dim, feature_map=None, eps=1e-6, k_size=None):
        super(linear_cla, self).__init__()
        query_dimensions = input_dim
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        # self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        
    #     self.reset_params()
        
    # def reset_params(self):
    #     nn.init.kaiming_normal_(self.Wv, mode='fan_out', nonlinearity='relu')

    def forward(self, x, s, z):
        """
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
        
        s <-- s + feature_map(K)V'
        z <-- z + feature_map(K)
        
        out <-- (feature_map(Q)s) / (feature_map(Q)z)
        """
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c] 
        K = self.Wk(y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        # V = V.view(b, 1, c*h*w) # V: [b, 1, chw]
        
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map()
        Q = self.feature_map.forward_queries(Q)
        K = self.feature_map.forward_keys(K)
        
        Q = Q.view(b, 1, c, 1)
        K = K.view(b, 1, c, 1)
        V = V.view(b, 1, c, h*w) # c heads (groups)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        # let d=hw, t=1, c=1, g=c
        KV = torch.einsum('btgc, btgd -> bgcd', K, V) # [b, c, 1, hw]  [b, g, c/g, c/ghw]
        s = s + KV

        # Compute the normalizer
        z = z + K # [b, 1, c, 1]
        QZ = 1.0 / torch.einsum('btgc, btgc -> btg', Q, z+self.eps) # [b, 1, c]
        # Finally compute and return the new values
        out = torch.einsum("btgc, bgcd, btg -> btgd", Q, s, QZ) # [b, 1, c, hw]
        out = out.contiguous().view(b, c, h, w)
        
        return out, s, z
    
    
class linear_gla(Module):
    '''
    Linear Groupwise Layer Attention
    '''
    def __init__(self, input_dim, feature_map=None, eps=1e-6, k_size=None, 
                 groups=None, dim_pergroup=None):
        super(linear_gla, self).__init__()
        query_dimensions = input_dim
        
        if (groups == None) and (dim_pergroup == None):
            raise ValueError("arguments groups and dim_pergroup cannot be None at the same time !")
        elif dim_pergroup != None:
            groups = int(input_dim / dim_pergroup)
        else:
            groups = groups
        self.groups = groups
        self.dim_pergroup = dim_pergroup
        
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        # self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        
    #     self.reset_params()
        
    # def reset_params(self):
    #     nn.init.kaiming_normal_(self.Wv, mode='fan_out', nonlinearity='relu')

    def forward(self, x, s, z):
        """
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
        
        s <-- s + feature_map(K)V'
        z <-- z + feature_map(K)
        
        out <-- (feature_map(Q)s) / (feature_map(Q)z)
        """
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c] 
        K = self.Wk(y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        V = V.view(b, 1, c, h*w) # V: [b, 1, c, hw]
        
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map()
        Q = self.feature_map.forward_queries(Q)
        K = self.feature_map.forward_keys(K)
        
        g = self.groups
        Q = Q.view(b, 1, g, int(c/g))
        K = K.view(b, 1, g, int(c/g))
        V = V.view(b, 1, g, int(c/g), h*w) # g heads (groups)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        # let d=hw, t=1, c=c/g (Q, K), g=g, s=c/g (V)
        KV = torch.einsum('btgc, btgsd -> bgcsd', K, V) # [b, g, c/g, c/ghw], [b, g, c/g, c/g, hw]
        s = s + KV

        # Compute the normalizer
        z = z + K # [b, 1, g, c/g]
        QZ = 1.0 / torch.einsum('btgc, btgc -> btg', Q, z+self.eps) # [b, 1, g]
        # Finally compute and return the new values
        out = torch.einsum("btgc, bgcsd, btg -> btgsd", Q, s, QZ) # [b, 1, g, c/g, hw]
        out = out.contiguous().view(b, 1, c, h*w) # [b, 1, c, hw]
        out = out.view(b, c, h, w)
        
        return out, s, z
    

