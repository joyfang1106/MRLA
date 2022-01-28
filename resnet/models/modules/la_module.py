import torch
from torch import nn
from torch.nn.parameter import Parameter
from math import sqrt
from math import log

 
class mla_layer(nn.Module):
    """
    layer attention module v7: groupwise operation of v2
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
    def __init__(self, input_dim, groups=None, dim_pergroup=None, k_size=None):
        super(mla_layer, self).__init__()
        self.input_dim = input_dim
        
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
        self.k_size = k_size
    
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
  
