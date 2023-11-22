import os, sys
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.abspath('.'))
import config as cfg
import cv2
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, cfg, data_path, train=True):
        self.data = cfg.DATA
        data = datasets.MNIST(root=data_path, train=train)
        if train:
            self.train_data, self.validation_data = self.train_vali_split(data)
        else:
            self.test_data = data
    
    def train_vali_split(self, data):
        train_data_len = int(len(data) * self.data['test_vali_split'])
        return random_split(data, [train_data_len, len(data)-train_data_len])
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        data, label = self.train_data[index]
        return data, label


class CustomDataLoader(DataLoader):
    def __init__(self, cfg, path='.'):
        self.cfg = cfg
        self.train_data = CustomDataset(cfg=cfg, data_path=path, train=True)
        self.test_data = CustomDataset(cfg=cfg, data_path=path, train=False)
    
    def __len__(self):
        return len(self.train_data)
    
    def get_items(self, index, index_end=False):
        if not index_end:
            data, label = self.train_data[index]
        else:
            data, label = [], []
            for idx in range(index, index_end):
                d, l = self.train_data[idx]
                data.append(d)
                label.append(l)
        return data, label


def get_dataloader(cfg, path='.'):
    cur_path = os.path.abspath(path)
    if not os.path.exists(os.path.join(cur_path, 'MNIST')):
        datasets.MNIST(cur_path, train=True, download=True)
        datasets.MNIST(cur_path, train=False, download=True)
    custom_dataloader = CustomDataLoader(cfg=cfg, path=path)
    return custom_dataloader


def test_dataloader(cfg, dataloader):
    for i in range(10):
        img, label = dataloader.train_data[i]
        img = np.array(img)
        cv2.imshow(str(label), img)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    dataloader = get_dataloader(cfg=cfg)
    test_dataloader(cfg=cfg, dataloader=dataloader)
