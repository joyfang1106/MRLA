import torch
from torch import nn
from torch.nn.parameter import Parameter
from math import sqrt
from math import log
from einops import rearrange



class mrla_base_layer(nn.Module):
    """
    multi-head layer attention module: MRLA-base
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
    def __init__(self, input_dim, heads=None, dim_perhead=None, k_size=None, init_cell=False):
        super(mrla_base_layer, self).__init__()
        self.input_dim = input_dim
        self.init_cell = init_cell
        
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, prev_K, prev_V):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c] 
        k = self.Wk(y) # k: [b, 1, c]
        v = self.Wv(x) # v: [b, c, h, w]
        
        if self.init_cell:
            K = k # K: [b, 1, c]
            V = v.unsqueeze(1) # V: [b, 1, c, h, w]
        else:        
            K = torch.cat((prev_K, k), dim=1) # K: [b, t, c]
            V = torch.cat((prev_V, v.unsqueeze(1)), dim=1) # V: [b, t, c, h, w]
        
        output_K = K
        output_V = V
      
        Q = Q.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        K = rearrange(K, 'b t (g d) -> b g t d', b=b, g=self.heads, d=int(c/self.heads)) # [b, g, t, c/g]
        V = rearrange(V, 'b t (g d) h w -> b g t d h w', g=self.heads, d=int(c/self.heads)) # [b, g, t, c/g, h, w]
        
        atten = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        # atten.size() # [b, g, 1, t]
        
        atten = self.softmax(atten)
        V = rearrange(V, 'b g t d h w -> b g t (d h w)')
    
        # output = atten @ V # [b g 1 (c/g h w)]
        output = torch.einsum('bgit, bgtj -> bgij', atten, V) 
        output = output.unsqueeze(2).reshape(b, c, h, w)
        
        return output, output_K, output_V
    
