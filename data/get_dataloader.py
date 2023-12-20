import os, sys
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.abspath('.'))
import config as cfg
import cv2
from PIL import Image
import numpy as np
import pickle
import struct

class CustomDataset(Dataset):
    def __init__(self, cfg, train=True, data_path='.'):
        self.cfg = cfg
        data_loda_func = {'MNIST':self._load_mnist}
        self.data = data_loda_func[cfg['type']](data_path)
    
    def _load_mnist(self, data_path):
        files = os.listdir(data_path)
        img_array, label_array = [], []
        for file in files:
            file = os.path.join(data_path, file)
            with open(file, 'rb') as f:
                zero, data_type, dims = struct.unpack('>HBB', f.read(4))
                shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
                if 'images' in file:
                    img_array.extend(np.frombuffer(f.read(), dtype=np.uint8).reshape(shape))
                else:
                    label_array.extend(np.frombuffer(f.read(), dtype=np.uint8).reshape(shape))
        
        data = []
        for img, label in zip(img_array, label_array):
            data.append([img, label])
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, label = self.data[index]
        return data, label

def get_dataloader(cfg, train=True, path='.'):
    if cfg['type'] == 'MNIST' and not os.path.exists(os.path.join(path, 'MNIST')):
        datasets.MNIST(path, train=train, download=True)
    path = os.path.join(path, cfg['type'])
    customDataset = CustomDataset(cfg, train, path)
    custom_dataloader = DataLoader(dataset=customDataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])
    return custom_dataloader


def test_dataloader(cfg):
    train_dataloader = get_dataloader(cfg=cfg, train=True, path='C:/Users/ghost/workspace/DeepLearning_SimpleCNN')
    test_dataloader = get_dataloader(cfg=cfg, train=False, path='C:/Users/ghost/workspace/DeepLearning_SimpleCNN')
    imgs, labels = next(iter(test_dataloader))
    for img, label in zip(imgs, labels):
        img = img.detach().cpu().numpy()
        print(label)
        cv2.imshow('asdf', img)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_dataloader(cfg.DATA)
