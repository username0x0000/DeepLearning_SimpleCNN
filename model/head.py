from torch import nn


class MlpClassifier(nn.Module):
    pass


class ResNet50_head(nn.Module):
    def __init__(self, cfg): # cfg.MODEL['ResNet50']['head']
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg['input_feature'], cfg['class_num'])

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Test_head(nn.Module):
    def __init__(self, cfg): # cfg.MODEL['ResNet50']['head']
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg['input_feature'], cfg['class_num'])

    def forward(self, x):
        print(x.shape)
        return x