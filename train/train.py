import sys
sys.path.append('./model')
sys.path.append('.')
sys.path.append('./data')
import config as cfg
import torch
from torch import nn
from torch import optim
from backbone import *
from head import *
from get_model import *
from get_dataloader import *
from trainer import *
from validatater import *


def train_main(pretrained=False):
    if pretrained:
        model = torch.load('model.pth')
        loss = get_loss(cfg.LOSS)
    else:
        train_dataloader = get_dataloader(cfg=cfg.DATA)
        model = get_model(cfg.MODEL, 'ResNet50')
        loss = get_loss(cfg.LOSS)
        optimizer = get_optimizer(cfg.OPTIMIZER, model)

        trainer = Trainer(model, loss, optimizer, cfg.TRAIN)
        trainer.train(train_dataloader)
        torch.save(model, 'model.pth')

    test_dataloader = get_dataloader(cfg=cfg.DATA, train=False)
    validater = Validatater(model, loss)
    validater.eval(test_dataloader)


def get_loss(cfg):
    if cfg['type'] == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()

    return loss


def get_optimizer(cfg, model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


if __name__ == '__main__':
    train_main(True)
