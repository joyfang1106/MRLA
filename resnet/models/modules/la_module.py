from einops import rearrange
import torch
from torch import nn, squeeze
from torch.nn.parameter import Parameter
from math import sqrt
from math import log

"""
build a layer attention module (eq.4. version)
"""

class la_layer(nn.Module):
    """
    Input:
        x: [b, c, h, w] # t-th layer's features
        ctx: [b, t, c, h, w] # up to t-th layer's features
    Output:
        x: [b, c, h, w] # after layer attention
    
    Operations:
        Wq, Wk: GAP + conv1d
        Wv: conv2d
        Q: [b, 1, c]
        K: [b, t, c]
        V: [b, t, c, h, w]
    """
    def __init__(self, input_dim, layer_idx, heads=None, dim_perhead=None, k_size=None):
        super(la_layer, self).__init__()

        if (heads == None) and (dim_perhead == None):
            raise ValueError("arguments heads and dim_perhead cannot be None at the same time !")
        elif dim_perhead != None:
            heads = int(input_dim / dim_perhead)
        else:
            heads = heads
        self.heads = heads

        if k_size == None:
            s = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = s if s % 2 else s+1
        self.k_size = k_size

        self.t = layer_idx # t-th layer

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        # self.Wk = nn.Conv1d(self.t, self.t, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False)
        self._norm_fact = 1 / sqrt(input_dim / heads)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, ctx):
        b, t, c, h, w = ctx.size()
        # feature descriptor on the global spatial information
        q = self.avg_pool(x) # [b, c, 1, 1]
        q = q.squeeze(-1).transpose(-1, -2) # [b, 1, c]

        k = rearrange(ctx, 'b t c h w -> (b t) c h w')
        k = self.avg_pool(k).squeeze(-1).transpose(-1, -2) # [bt, c, 1, 1] --> [bt, 1, c]
        # k = rearrange(k, '(b t) c -> b t c', b=b) # [b, t, c]
        v = rearrange(ctx, 'b t c h w -> (b t) c h w')

        Q = self.Wq(q) # Q: [b, 1, c]
        K = self.Wk(k).squeeze(-2)  # [bt, c]  
        K = rearrange(K, '(b t) c -> b t c', b=b) # K: [b, t, c]
        V = self.Wv(v).view(b, t, c, h, w) # V: [b, t, c, h, w]

        Q = Q.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        K = K.view(b, self.heads, t, int(c/self.heads)) # [b, g, t, c/g]
        V = V.view(b, self.heads, t, int(c/self.heads), h, w) # [b, g, t, c/g, h, w]
        attn = torch.einsum('... i d, ... t d -> ... i t', Q, K) * self._norm_fact # [b, g, 1, t]
        attn = self.softmax(attn) # [b, g, 1, t]
        output = torch.einsum('bgit, bgtdhw -> bgidhw', attn, V) # [b, g, 1, c/g, h, w]
        output = output.view(b, c, h, w)     
        
        return output

if __name__ == "__main__":
    import torch
    bsz = 2
    c = 512
    h = 256
    w = 256
    t = 3
    x = torch.randn(bsz, c, h, w)
    ctx = torch.randn(bsz, t, c, h, w)

    la = la_layer(c, t, heads=8)

    out = la(x, ctx)

    print('out.shape', out.shape)




