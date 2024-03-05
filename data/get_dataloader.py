import os, sys
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import struct
import json
import PIL


class CustomDataset(Dataset):
    def __init__(self, cfg, train=True, data_path='.'):
        self.cfg = cfg
        self.train = train
        data_loda_func = {'MNIST': self._load_mnist, 'COCO': self._load_coco}
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

    def _load_coco(self, data_path):
        if self.train:
            json_data_path = os.path.join(data_path, 'annotations', 'instances_train2017.json')
            with open(json_data_path) as js_file:
                self.json = json.load(js_file)

            img_dict = {}
            img_path = os.path.join(data_path, 'train2017')
            for img in self.json['images']:
                img_dict[img['id']] = {'img': os.path.join(img_path, img['file_name']), 'bbox': [], 'category': []}
            for ann in self.json['annotations']:
                img_dict[ann['image_id']]['bbox'].append([int(p) for p in ann['bbox']])
                img_dict[ann['image_id']]['category'].append(ann['category_id'])
            data = [img_dict[k] for k in img_dict.keys()]
            return data

        else:
            json_data_path = os.path.join(data_path, 'annotations', 'instances_test2017.json')
            with open(json_data_path) as js_file:
                self.json = json.load(js_file)

            data = []
            img_dict = {}
            img_path = os.path.join(data_path, 'test2017')
            for img in self.json['images']:
                img_dict[img['id']] = {'img': os.path.join(img_path, img['file_name']), 'bbox': [], 'category': []}
            for ann in self.json['annotations']:
                img_dict[ann['image_id']]['bbox'].append([int(p) for p in ann['bbox']])
                img_dict[ann['image_id']]['category'].append(ann['category_id'])
            data = [img_dict[k] for k in img_dict.keys()]
            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = self.data[index]
        return data, label


class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.idx = -batch_size
        self.batch_size = batch_size

    def __getitem__(self, _idx):
        self.idx += self.batch_size
        return self.dataset[self.idx: self.idx + 64]


def get_dataloader(cfg, train=True, path='.'):
    if cfg['type'] == 'MNIST' and not os.path.exists(os.path.join(path, 'MNIST')):
        datasets.MNIST(path, train=train, download=True)
    path = os.path.join(path, cfg['type'])
    customDataset = CustomDataset(cfg, train, path)
    custom_dataloader = CustomDataLoader(dataset=customDataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])
    return custom_dataloader


def print_cat(category):
    categories = {0: '__background__', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                  7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign',
                  13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
                  20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella',
                  27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard',
                  33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard',
                  38: 'surfboard', 39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork',
                  44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange',
                  51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair',
                  58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop',
                  65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven',
                  71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase',
                  77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush', 81: 'sink',
                  82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
                  89: 'hair drier', 90: 'toothbrush'}
    category = categories[category]
    print(category)


def test_dataloader(cfg):
    train_dataloader = get_dataloader(cfg=cfg, train=True, path='.')
    # test_dataloader = get_dataloader(cfg=cfg, train=False, path='.')
    datas = next(iter(train_dataloader))
    for data in datas:
        img = cv2.imread(data['img'])
        for bbox, cat in zip(data['bbox'], data['category']):
            print(cat)
            pt1, pt2, pt3, pt4 = bbox
            cv2.rectangle(img, (pt1, pt2), [pt1+pt3, pt2+pt4], color=[255, 0, 0], thickness=3)
            cv2.imshow('asdf', img)
            k = cv2.waitKey()
            if k == 120:
                cv2.destroyAllWindows()
                exit()
    # for datas in train_dataloader:
    #     for data in datas:
    #         img = cv2.imread(data['img'])
    #         for bbox, cat in zip(data['bbox'], data['category']):
    #             print_cat(cat)
    #             pt1, pt2, pt3, pt4 = bbox
    #             cv2.rectangle(img, (pt1, pt2), [pt1 + pt3, pt2 + pt4], color=[255, 0, 0], thickness=3)
    #             cv2.imshow('asdf', img)
    #             k = cv2.waitKey()
    #             if k == 120:
    #                 cv2.destroyAllWindows()
    #                 exit()


if __name__ == '__main__':
    sys.path.append(os.path.abspath('..'))
    import config as cfg

    test_dataloader(cfg.DATA)
