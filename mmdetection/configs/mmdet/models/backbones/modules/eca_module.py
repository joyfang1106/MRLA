import torch
from torch import nn
from torch.nn.parameter import Parameter
from math import log


# from .modules.eca_module import eca_layer
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        source: https://github.com/BangguWu/ECANet
    """
    def __init__(self, channel, k_size=None):
        super(eca_layer, self).__init__()
        if k_size == None:
            t = int(abs((log(channel, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # y = torch.randn(4, 3, 1, 1)  # print('y :', y.size())
        # y.squeeze(-1) y : torch.Size([4, 3, 1])
        # y.transpose(-1, -2) y : torch.Size([4, 1, 3])
        # --> y : torch.Size([4, 3, 1]) --> torch.Size([4, 3, 1, 1])
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # similar to first .view(b, c) and then .view(b, c, 1, 1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
    