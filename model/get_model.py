import torch
import sys
sys.path.append('.')
import config as cfg
from backbone import *
from head import *


class Model(nn.Module):
    def __init__(self, cfg, pick_model): # cfg.MODEL
        if pick_model == 'ResNet50':
            self.backbone = ResNet50_backbone(cfg['ResNet50']['backbone'])
            self.head = ResNet50_head(cfg['ResNet50']['head'])
    
    def forward(self, x):
        x = self.backbone.forward(x)
        x = self.head.forward(x)
        return x


def get_model(cfg, pick_model):
    return Model(cfg, pick_model)
    """
    cfg에 따라 backbone, head를 조합하여 model 리턴
    """



def test_model():
    """
    get_model()로 model을 만들고 임의의 이미지를 넣었을 때 에러 없이 돌아가는지 확인
    backbone과 head의 호환성 확인
    head의 출력의 shape 확인
    이미지는 그냥 shape만 맞춘 torch.Tensor 만들어 사용
    """
    pass


if __name__ == '__main__':
    model = get_model(cfg.MODEL, 'ResNet50')
    print('model load')
