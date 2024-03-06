import torch
from torch import nn
import sys, os
sys.path.append(os.path.abspath('..'))
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


class VGG_backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.layer1 = self._make_layer([64, 64])
        self.layer2 = self._make_layer([128, 128])
        self.layer3 = self._make_layer([256, 256, 256, 256])
        self.layer4 = self._make_layer([512, 512, 512, 512])
        self.layer5 = self._make_layer([512, 512, 512, 512])
    
    def _make_layer(self, channels):
        conv64 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        conv128 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=False)
        conv256 = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1, bias=False)
        conv512 = nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        layers = []
        
        for channel in channels:
            if channel == 64:
                layers.append(conv64)
            elif channel == 128:
                layers.append(conv128)
            elif channel == 256:
                layers.append(conv256)
            elif channel == 512:
                layers.append(conv512)
        
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        layers.append(maxpool)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        result = self.layer5(x)
        return result


class Darknet53_ResBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        half_input_channels = int(input_channels)
        self.conv1 = nn.Conv2d(input_channels, half_input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(half_input_channels)
        self.conv2 = nn.Conv2d(half_input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        start = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.activation(out+start)


class Darknet53(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = [] # [neck_output, layer]
        self.layers.append([False, nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)])
        self.layers.append([False, nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)])
        self.layers.append([False, nn.MaxPool2d(kernel_size=2, stride=2)])

        self.layers.append([False, self._make_layer(64, 1)])
        self.layers.append([False, nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)])
        self.layers.append([False, nn.MaxPool2d(kernel_size=2, stride=2)])

        self.layers.append([True, self._make_layer(128, 2)])
        self.layers.append([False, nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)])
        self.layers.append([False, nn.MaxPool2d(kernel_size=2, stride=2)])

        self.layers.append([True, self._make_layer(256, 8)])
        self.layers.append([False, nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)])
        self.layers.append([False, nn.MaxPool2d(kernel_size=2, stride=2)])

        self.layers.append([True, self._make_layer(512, 8)])
        self.layers.append([False, nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)])
        self.layers.append([False, nn.MaxPool2d(kernel_size=2, stride=2)])

        self.layers.append([True, self._make_layer(1024, 4)])
    
    def _make_layer(self, input_channels, layer_num):
        layers = []
        for num in range(layer_num):
            layers.append(Darknet53_ResBlock(input_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        for neck_output, layer in self.layers:
            x = layer(x)
            if neck_output:
                output.appned(x)
        return output


if __name__ == '__main__':
    # res = ResNet50_backbone(cfg=cfg.MODEL['ResNet50']['backbone'])
    # print(res)
    # print()
    # print('--'*30)
    # print()
    darknet = Darknet53(cfg=1)
    print(darknet)
