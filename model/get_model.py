import torch
import sys
sys.path.append('.')
sys.path.append('data')
import config as cfg
from backbone import *
from head import *
from get_dataloader import *
# import cv2
import numpy as np


class Model(nn.Module):
    def __init__(self, cfg, pick_model): # cfg.MODEL
        super().__init__()
        if pick_model == 'ResNet50':
            self.backbone = ResNet50_backbone(cfg['ResNet50']['backbone'])
            self.head = MlpClassifier(cfg['ResNet50']['head'])
        elif pick_model == 'VGG':
            self.backbone = VGG_backbone(cfg['VGG'])
            self.head = MlpClassifier(cfg['VGG']['head'])
        else:
            self.backbone = ResNet50_backbone(cfg['ResNet50']['backbone'])
            self.head = Test_head(cfg['ResNet50']['head'])
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def get_model(cfg, pick_model):
    return Model(cfg, pick_model)
    """
    cfg에 따라 backbone, head를 조합하여 model 리턴
    """


def test_model():
    model = get_model(cfg.MODEL, '_')
    img = torch.randint(low=0, high=255, size=(32, 1, 28, 28), dtype=torch.float)
    model(img)
    """
    get_model()로 model을 만들고 임의의 이미지를 넣었을 때 에러 없이 돌아가는지 확인
    backbone과 head의 호환성 확인
    head의 출력의 shape 확인
    이미지는 그냥 shape만 맞춘 torch.Tensor 만들어 사용
    """


if __name__ == '__main__':
    # model = get_model(cfg.MODEL, 'ResNet50')
    # print('model load')
    test_model()
