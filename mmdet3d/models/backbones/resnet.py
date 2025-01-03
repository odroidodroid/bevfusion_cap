from typing import List, Tuple

import torch
# from mmcv.cnn.resnet import BasicBlock, Bottleneck, make_res_layer
from torch import nn
from mmdet.models import BACKBONES
# from mmcv.runner import load_checkpoint
import os
from collections import OrderedDict

save_dir = './'
os.makedirs(save_dir, exist_ok=True)
__all__ = ["CustomResNet"]

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResNetBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBottleneckBlock, self).__init__()
        
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('act', nn.ReLU(inplace=True))
        ]))
        
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('act', nn.ReLU(inplace=True))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels * 4))
        ]))
        
        self.downsample = downsample

        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity

        x = self.final_act(x)
        
        return x

@BACKBONES.register_module()
class CustomResNet(nn.Module):
    def __init__(self,
                 layers, 
                 out_indices=(0, 1, 2, 3),
                 pretrained = None
                 ):
        
        super(CustomResNet, self).__init__()
        self.out_indices = out_indices
        
        self.input_stem = nn.ModuleList([
            ConvLayer(3, 32, kernel_size=3, stride=2, padding=1), 
            ConvLayer(32, 64, kernel_size=3, stride=1, padding=1)
        ])
        
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.blocks = nn.ModuleList()
        
        in_channels = 64
        for i, num_blocks in enumerate(layers):
            stride = 1 if i == 0 else 2
            out_channels = 64 * (2 ** i)
            
            for j in range(num_blocks):
                downsample = None
                # Apply downsample only in the first block of each stage where output channels change
                if j == 0 and (stride != 1 or in_channels != out_channels * 4):
                    downsample = nn.Sequential(OrderedDict([
                        ('avg_pool', nn.AvgPool2d(kernel_size=1, stride=stride)),
                        ('conv', nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)),
                        ('bn', nn.BatchNorm2d(out_channels * 4))
                    ]))
                self.blocks.append(ResNetBottleneckBlock(in_channels, out_channels, stride if j == 0 else 1, downsample))
                in_channels = out_channels * 4

    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained)
            print(checkpoint)
            self.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Loaded pretrained weights from {pretrained}")
        elif pretrained:
            print("Pretrained argument is not a valid path.")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)

        x = self.max_pooling(x)
        
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)


model = CustomResNet(layers = [4, 4, 6, 4])
f = open(os.path.join(save_dir, f'ResNet.txt'), 'w')
print(model, file=f)
f.close()


