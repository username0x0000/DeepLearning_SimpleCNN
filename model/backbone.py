import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, cfg):
        input_channels, output_channels, stride = cfg['input channels'], cfg['output channels'], cfg['stride']
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


class ResNet50(nn.Module):
    pass

