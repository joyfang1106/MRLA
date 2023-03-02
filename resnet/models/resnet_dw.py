'''
directly add a dw conv after each layer
'''
import torch
import torch.nn as nn
from models.utils.drop import DropPath

from .modules.eca_module import eca_layer # not use
from .modules.se_module import se_layer # not use


__all__ = ['ResNet_dw', 
           'resnet50_dw'
           ]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck_dw(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 SE=False, ECA=False, groups=1,
                 base_width=64, dilation=1, 
                 norm_layer=nn.BatchNorm2d, drop_path=0.0):
        super(Bottleneck_dw, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
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

        self.se = None
        if SE:
            self.se = se_layer(planes * self.expansion, reduction=16)
        
        self.eca = None
        if ECA:
            self.eca = eca_layer(planes * self.expansion)
            
        self.dwconv = nn.Conv2d(planes * self.expansion, planes * self.expansion, kernel_size=3, stride=1, padding=1, groups=planes * self.expansion, bias=False) 
        self.bn_dw = norm_layer(planes * self.expansion)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.se != None:
            out = self.se(out)
            
        if self.eca != None:
            out = self.eca(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        out = out + self.drop_path(self.bn_dw(self.dwconv(out)))

        return out
    
    

class ResNet_dw(nn.Module):

    def __init__(self, block, 
                layers, 
                num_classes=1000, 
                SE=False, 
                ECA=False, 
                zero_init_last_bn=True, #zero_init_residual=False,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                norm_layer=nn.BatchNorm2d, 
                drop_rate=0.0, 
                drop_path=0.0):
        super(ResNet_dw, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.drop_path = drop_path
        self.SE = SE
        self.ECA = ECA
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_last_bn:
            for m in self.modules():
                if isinstance(m, Bottleneck_dw):
                    nn.init.constant_(m.bn3.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
                            SE=self.SE, ECA=self.ECA, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, drop_path=self.drop_path))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                SE=self.SE, ECA=self.ECA, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, drop_path=self.drop_path))
            
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # classifier head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)



def resnet50_dw(**kwargs):
    """ Constructs a ResNet-50 model.
    default: 
        num_classes=1000, SE=False, ECA=False
    ECA: a list of kernel sizes in ECA
    """
    print("Constructing resnet50_dw......")
    model = ResNet_dw(Bottleneck_dw, [3, 4, 6, 3], **kwargs)
    return model



