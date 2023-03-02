'''
Layer attention at equation (4)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.modules.la_module import la_layer
from models.modules.eca_module import eca_layer # not use
from models.modules.se_module import se_layer # not use
from models.utils.drop import DropPath, drop_path

__all__ = ['ResNet_la_eq4', 
           'resnet50_la_eq4', 
           'resnet101_la_eq4', 
           ]

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class LABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer_idx, dim_per_head=32,
                 stride=1, downsample=None, 
                 SE=False, ECA_size=None, 
                 groups=1, base_width=64, dilation=1, 
                 norm_layer=nn.BatchNorm2d, 
                 drop_path=0.0):
        super(LABottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
        
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

        # layer attention module
        self.la = la_layer(input_dim=planes * self.expansion, layer_idx=layer_idx,
                           dim_perhead=dim_per_head)
        self.bn_la = norm_layer(planes * self.expansion)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x, mem_x):

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
        ## mem_x is a list containing tensors with size [b, c, h, w]
        mem_x.append(out)
        ctx = torch.stack(mem_x, 1) # [b, t, c, h, w]
        out = self.bn_la(self.la(out, ctx))

        return out, mem_x



class ResNet_la_eq4(nn.Module):
    def __init__(self, block, 
                layers, 
                num_classes=1000, 
                #  la_channels=[256, 512, 1024, 2048], 
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
        super(ResNet_la_eq4, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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

         # stages = [None] * 4
        self.layer1 = self._make_layer(block, 64, layers[0], SE=SE, ECA_size=ECA[0])
        self.layer2 = self._make_layer(block, 128, layers[1], SE=SE, ECA_size=ECA[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], SE=SE, ECA_size=ECA[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], SE=SE, ECA_size=ECA[3], stride=2, dilate=replace_stride_with_dilation[2])
        # self.stages = nn.ModuleList(stages)
        
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
                if isinstance(m, LABottleneck):
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
        layers.append(block(self.inplanes, planes, 1, 32, stride, downsample, 
                            SE=SE, ECA_size=ECA_size, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, 
                            drop_path=self.drop_path))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, i+1, 32,
                                SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, 
                                drop_path=self.drop_path))
        
        return nn.Sequential(*layers)

    def forward_features(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, blk in enumerate(self.layer1):
            if i == 0:
                x, mem_x = blk(x, [])
            else:
                x, mem_x = blk(x, mem_x)
        
        for i, blk in enumerate(self.layer2):
            if i == 0:
                x, mem_x = blk(x, [])
            else:
                x, mem_x = blk(x, mem_x)

        for i, blk in enumerate(self.layer3):
            if i == 0:
                x, mem_x = blk(x, [])
            else:
                x, mem_x = blk(x, mem_x)
        
        for i, blk in enumerate(self.layer4):
            if i == 0:
                x, mem_x = blk(x, [])
            else:
                x, mem_x = blk(x, mem_x)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x
        
    

def resnet50_la_eq4(**kwargs):
    print("Constructing resnet50_la_eq4......")
    model = ResNet_la_eq4(LABottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101_la_eq4(**kwargs):
    print("Constructing resnet101_la_eq4......")
    model = ResNet_la_eq4(LABottleneck, [3, 4, 23, 3], **kwargs)
    return model



if __name__ == "__main__":
    import torch
    model = ResNet_la_eq4(LABottleneck, [3, 4, 6, 3], num_classes=1000, 
                #  la_channels=[256, 512, 1024, 2048], 
                SE=False, 
                ECA=None, 
                zero_init_last_bn=True, #zero_init_residual=False,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                norm_layer=nn.BatchNorm2d, 
                drop_rate=0.0, 
                drop_path=0.0)
    
    inp = torch.randn(2, 3, 256, 256)
    print(inp.shape)
    out = model(inp)

    print('out.shape', out.shape)

