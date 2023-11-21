import torch
from torch import nn
import sys
sys.path.append('.')
import config as cfg


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = nn.Sequential()
        
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        pass

    def forward(self, x):
        start = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        self.add_input_output(start, out)
        out = self.relu(out)
        return out
    
    def add_input_output(self, input_map, output_map):
        output_map += self.downsample(input_map)
        return output_map


class ResNet50_backbone(nn.Module):
    def __init__(self, cfg): # cfg.MODEL['ResNet51']['backbone']
        super().__init__()
        self.conv1 = nn.Conv2d(cfg['input_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(cfg['first_layer'])
        self.layer2 = self._make_layer(cfg['second_layer'])
        self.layer3 = self._make_layer(cfg['third_layer'])
        self.layer4 = self._make_layer(cfg['forth_layer'])

    def _make_layer(self, cfg):
        layers = []
        input_channels, output_channels, stride, layer_num = cfg['input_channels'], cfg['output_channels'], cfg['stride'], cfg['layer_num']
        layers.append(ResBlock(input_channels, output_channels, stride))
        for _ in range(1, layer_num):
            layers.append(ResBlock(output_channels, output_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        result = self.layer4(x)
        return result