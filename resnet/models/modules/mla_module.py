import torch
from torch import nn
from torch.nn.parameter import Parameter
from math import sqrt
from math import log


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

# torch.bmm(input, mat2, *, out=None)
# Performs a batch matrix-matrix product of matrices stored in input and mat2.
# input and mat2 must be 3-D tensors each containing the same number of matrices.
# If input is a (bxnxm) tensor, mat2 is a (bxmxp) tensor, out will be a (bxnxp) tensor.


class mla_layer(nn.Module):
    """
    multi-head layer attention module
    when heads = channels, channelwise (Q(K)' is then pointwise(channelwise) multiplication)
    
    Args:
        input_dim: input channel c (output channel is the same)
        heads: number of heads
        dim_perhead: channels per head
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
        super(mla_layer, self).__init__()
        self.input_dim = input_dim
        
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
        self.k_size = k_size
    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        self._norm_fact = 1 / sqrt(input_dim / heads)
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
        # Q = Q.chunk(self.heads, dim = -1) # a tuple of g * [b, 1, c/g]
        # K = K.chunk(self.heads, dim = -1)
        # V = V.chunk(self.heads, dim = 1) # a tuple of g * [b, c/g, h, w]
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
    
