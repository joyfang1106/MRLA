# This piece of code is developed based on
# https://github.com/pytorch/examples/tree/master/imagenet
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.eca_module import eca_layer
from .modules.se_module import se_layer


__all__ = ['ResNet', 'resnet50', 'resnet50_se', 'resnet50_eca',
           'resnet101', 'resnet101_se', 'resnet101_eca',
           'resnet152', 'resnet152_se', 'resnet152_eca',
           'resnext50_32x4d', 'resnext50_32x4d_se', 'resnext50_32x4d_eca',
           'resnext101_32x4d', 'resnext101_32x4d_se', 'resnext101_32x4d_eca'
           ]

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 SE=False, ECA_size=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(Bottleneck, self).__init__()
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
            self.se = se_layer(planes * self.expansion, reduction)
        
        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))

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

        return out
    
    

class ResNet(nn.Module):

    def __init__(self, block, 
                 layers, 
                 num_classes=1000, 
                 SE=False, 
                 ECA=None, 
                 zero_init_last_bn=True, #zero_init_residual=False,
                 groups=1, 
                 width_per_group=64, 
                 replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm2d, 
                 drop_rate=0.0):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        
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
        
        self.layer1 = self._make_layer(block, 64, layers[0], SE=SE, ECA_size=ECA[0])
        self.layer2 = self._make_layer(block, 128, layers[1], SE=SE, ECA_size=ECA[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], SE=SE, ECA_size=ECA[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], SE=SE, ECA_size=ECA[3], stride=2, dilate=replace_stride_with_dilation[2])
        
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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, SE, ECA_size, stride=1, dilate=False):
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
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            
        return nn.Sequential(*layers)

    def forward_features(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x



def resnet50(**kwargs):
    """ Constructs a ResNet-50 model.
    default: 
        num_classes=1000, SE=False, ECA=None
    ECA: a list of kernel sizes in ECA
    """
    print("Constructing resnet50......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet50_se(**kwargs):
    """ Constructs a ResNet-50_SE model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnet50_se......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], SE=True, **kwargs)
    return model

def resnet50_eca(k_size=[5, 5, 5, 7], **kwargs): # pretrained=False
    """Constructs a ResNet-50_ECA model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing resnet50_eca......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], ECA=k_size, **kwargs)
    return model


def resnet101(**kwargs):
    """ Constructs a ResNet-101 model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnet101......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet101_se(**kwargs):
    """ Constructs a ResNet-101_SE model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnet101_se......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], SE=True, **kwargs)
    return model

def resnet101_eca(k_size=[5, 5, 5, 7], **kwargs): 
    """Constructs a ResNet-101_ECA model.
    Args:
        k_size: Adaptive selection of kernel size
    """
    print("Constructing resnet101_eca......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], ECA=k_size, **kwargs)
    return model


def resnet152(**kwargs):
    """ Constructs a ResNet-152 model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnet152......")
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet152_se(**kwargs):
    """ Constructs a ResNet-152_SE model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnet152_se......")
    model = ResNet(Bottleneck, [3, 8, 36, 3], SE=True, **kwargs)
    return model

def resnet152_eca(k_size=[5, 5, 5, 7], **kwargs): 
    """Constructs a ResNet-152_ECA model.
    Args:
        k_size: Adaptive selection of kernel size
    """
    print("Constructing resnet152_eca......")
    model = ResNet(Bottleneck, [3, 8, 36, 3], ECA=k_size, **kwargs)
    return model


def resnext50_32x4d(**kwargs):
    """ Constructs a ResNeXt50_32x4d model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnext50_32x4d......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)
    return model

def resnext50_32x4d_se(**kwargs):
    """ Constructs a ResNeXt50_32x4d_SE model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnext50_32x4d_se......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], SE=True, groups=32, width_per_group=4, **kwargs)
    return model

def resnext50_32x4d_eca(k_size=[5, 5, 5, 7], **kwargs): 
    """Constructs a ResNeXt50_32x4d_ECA model.
    Args:
        k_size: Adaptive selection of kernel size
    """
    print("Constructing resnext50_32x4d_eca......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], ECA=k_size, groups=32, width_per_group=4, **kwargs)
    return model


def resnext101_32x4d(**kwargs):
    """ Constructs a ResNeXt101_32x4d model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnext101_32x4d......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=4, **kwargs)
    return model

def resnext101_32x4d_se(**kwargs):
    """ Constructs a ResNeXt101_32x4d_SE model.
    default: 
        num_classes=1000, SE=False, ECA=None
    """
    print("Constructing resnext101_32x4d_se......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], SE=True, groups=32, width_per_group=4, **kwargs)
    return model

def resnext101_32x4d_eca(k_size=[5, 5, 5, 7], **kwargs): 
    """Constructs a ResNeXt101_32x4d_ECA model.
    Args:
        k_size: Adaptive selection of kernel size
    """
    print("Constructing resnext101_32x4d_eca......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], ECA=k_size, groups=32, width_per_group=4, **kwargs)
    return model

