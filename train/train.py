import sys
sys.path.append('../model')
sys.path.append('..')
sys.path.append('../data')
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
        accuracy, epoch = 0, 0
        while accuracy < 0.8 or epoch < 1000:
            train_dataloader = get_dataloader(cfg=cfg.DATA, path='../data')
            model = get_model(cfg.MODEL, 'ResNet50')
            loss = get_loss(cfg.LOSS)
            optimizer = get_optimizer(cfg.OPTIMIZER, model)

            trainer = Trainer(model, loss, optimizer, cfg.TRAIN)
            trainer.train(train_dataloader)
            torch.save(model, 'model1.pth')

            test_dataloader = get_dataloader(cfg=cfg.DATA, train=False, path='../data')
            validater = Validatater(model, loss)
            accuracy = validater.eval(test_dataloader)
            epoch += 10


def get_loss(cfg):
    if cfg['type'] == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    return loss


def get_optimizer(cfg, model):
    return optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


if __name__ == '__main__':
    train_main(False)
