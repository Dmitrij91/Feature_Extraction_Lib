
import torch
import torch.nn as nn
from functools import partial

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)  


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,Features = False, activation='relu'):
        super().__init__()
        self.Features = Features
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=1,Hdim = 1, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        layers = []

        if Hdim != 1:
            layers.append(nn.Conv2d(Hdim, in_channels, kernel_size=3, stride=1, padding=1))

        layers.extend([
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])

        self.gate = nn.Sequential(*layers)
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes,Features, dropout=0.0):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if Features:
            self.decoder = nn.Sequential(nn.Dropout(p=1.0-dropout),
                nn.Identity(in_features)
                                        )   
        else:
            self.decoder = nn.Sequential(
                nn.Dropout(p=1.0-dropout),
                nn.Linear(in_features, n_classes)
            )

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes,Features, dropout=0.0,Hdim = 1, *args, **kwargs):
        super().__init__()
        self.Features = Features
        self.encoder = ResNetEncoder(in_channels,Hdim = Hdim, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes,Features, dropout)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet(size, in_channels, n_classes,Feat, *args, **kwargs):
    layer_depths = {
        5:   [1, 1, 1, 1],
        6:   [1, 1, 1, 1],
        7:   [1, 1, 1, 1],
        8:   [1, 1, 1, 1],
        9:   [1, 1, 1, 1],
        10:  [1, 1, 1, 1],
        18:  [2, 2, 2, 2], 
        34:  [3, 4, 6, 3],
        50:  [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    blocks = {
        5:   ResNetBasicBlock,
        6:   ResNetBasicBlock,
        7:   ResNetBasicBlock,
        8:   ResNetBasicBlock,
        9:   ResNetBasicBlock,
        10:  ResNetBasicBlock,
        18:  ResNetBasicBlock,
        34:  ResNetBasicBlock,
        50:  ResNetBottleNeckBlock,
        101: ResNetBottleNeckBlock,
        152: ResNetBottleNeckBlock
    }
    channel_dims = {
        5:   [2, 4, 8, 16],
        6:   [4, 8, 16, 32],
        7:   [8, 16, 32, 64],
        8:   [16, 32, 64, 128],
        9:   [32, 64, 128, 256],
        10:  [64, 128, 256, 512],
        18:  [64, 128, 256, 512],
        34:  [64, 128, 256, 512],
        50:  [64, 128, 256, 512],
        101: [64, 128, 256, 512],
        152: [64, 128, 256, 512]
    }
    assert layer_depths.keys() == blocks.keys()
    assert size in layer_depths.keys()
    return ResNet(in_channels, n_classes,Features = Feat, block=blocks[size], blocks_sizes=channel_dims[size], depths=layer_depths[size], *args, **kwargs)
