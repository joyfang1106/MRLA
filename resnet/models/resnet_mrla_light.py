'''
# multi-head recurrent layer attention, light-weighted version at equation (6)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.mrla_light_module import mrla_light_layer
from models.modules.eca_module import eca_layer
from models.modules.se_module import se_layer
from models.utils.drop import DropPath


__all__ = ['ResNet_mrlal', 
           'resnet50_mrlal', 
           'resnet101_mrlal', 
           'resnet152_mrlal'
           ]



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class mrla_module(nn.Module):
    dim_perhead = 32
    
    def __init__(self, input_dim):
        super(mrla_module, self).__init__()
        self.mrla = mrla_light_layer(input_dim=input_dim, dim_perhead=self.dim_perhead)
        self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  
        
    def forward(self, xt, ot_1):
        attn_t = self.mrla(xt)
        out = attn_t + self.lambda_t.expand_as(ot_1) * ot_1 
        return out



class MRLA_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, 
                 stride=1, downsample=None, 
                 SE=False, ECA_size=None, 
                 groups=1, base_width=64, dilation=1, 
                 norm_layer=nn.BatchNorm2d, drop_path=0.0):
        super(MRLA_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm 
            # # support for multi gpus
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        
        # channel attention modules
        self.se = None
        if SE:
            self.se = se_layer(planes * self.expansion, reduction=16)
        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))
            
        # mrla module
        self.mrla = mrla_module(input_dim=planes * self.expansion)
        self.bn_mrla = norm_layer(planes * self.expansion)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # nn.Identity() is placeholder
        
    def forward(self, x):
        identity = x
        
        # res block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # channel attention
        if self.se != None:
            out = self.se(out)
        if self.eca != None:
            out = self.eca(out) 
        # downsampling for short cut    
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity 
        out = self.relu(out)
        # layer attention
        out = out + self.drop_path(self.bn_mrla(self.mrla(out, identity)))
        
        return out
    
    

class ResNet_mrlal(nn.Module):
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
                drop_rate=0.0, 
                drop_path=0.0
                ):
        super(ResNet_mrlal, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.drop_path = drop_path
        
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
            # elif isinstance(m, (nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_last_bn:
            for m in self.modules():
                if isinstance(m, MRLA_Bottleneck):
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
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, drop_path=self.drop_path))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, drop_path=self.drop_path))
            
        return nn.Sequential(*layers)
        # return nn.ModuleList(layers)

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



def resnet50_mrlal(**kwargs):
    print("Constructing resnet50_mrla-light......")
    model = ResNet_mrlal(MRLA_Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101_mrlal(**kwargs):
    print("Constructing resnet101_mrla-light......")
    model = ResNet_mrlal(MRLA_Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

