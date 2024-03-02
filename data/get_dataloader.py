import os, sys
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import struct
import json
import PIL
from pycocotools.coco import COCO
sys.path.append(os.path.abspath('..'))
import config as cfg

class CustomDataset(Dataset):
    def __init__(self, cfg, train=True, data_path='.'):
        self.cfg = cfg
        self.train = train
        data_loda_func = {'MNIST':self._load_mnist, 'COCO':self._load_coco}
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
            train_val_info_path = os.path.join(data_path, 'annotations_trainval2017/annotations')
            train_image_path = os.path.join(data_path, 'train2017')
            val_image_path = os.path.join(data_path, 'val2017')

            val_info = COCO(os.path.join(train_val_info_path, 'instances_val2017.json'))
            train_info = COCO(os.path.join(train_val_info_path, 'instances_train2017.json'))

            train_image_list = []
            for coco_id in train_info.anns:
                ann = train_info.loadAnns(coco_id)[0]
                img_id = ann['image_id']
                category_id = ann['category_id']
                bbox = ann['bbox']
                img_path = os.path.join(train_image_path, train_info.loadImgs(img_id)[0]['file_name'])
                train_image_list.append([img_path, {'category':category_id, 'bbox':bbox}])

            val_image_list = []
            for coco_id in val_info.anns:
                ann = val_info.loadAnns(coco_id)[0]
                img_id = ann['image_id']
                category_id = ann['category_id']
                bbox = ann['bbox']
                img_path = os.path.join(val_image_path, val_info.loadImgs(img_id)[0]['file_name'])
                val_image_list.append([img_path, {'category':category_id, 'bbox':bbox}])
            return train_image_list + val_image_list

        else:
            test_image_path = os.path.join(data_path, 'test2017')
            # test_info = os.path.join(data_path, 'image_info_test2017/annotations/')
            # test_info = COCO(test_info)
            # train_info = COCO(os.path.join(train_val_info_path, 'instances_train2017.json'))
            #
            # train_image_list = []
            # for coco_id in train_info.anns:
            #     ann = train_info.loadAnns(coco_id)[0]
            #     img_id = ann['image_id']
            #     category_id = train_info.loadCats(ann['category_id'])
            #     bbox = ann['bbox']
            #     img_path = os.path.join(train_image_path, train_info.loadImgs(img_id)[0]['file_name'])
            #     train_image_list.append([[img_path], {'category':category_id, 'bbox':bbox}])
    
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
    train_dataloader = get_dataloader(cfg=cfg, train=True, path='.')
    test_dataloader = get_dataloader(cfg=cfg, train=False, path='.')

    imgs, labels = next(iter(train_dataloader))
    idx = 0
    for img in imgs:
        category = labels['category'][idx]
        pt1, pt2, pt3, pt4 = labels['bbox'][0][idx], labels['bbox'][1][idx], labels['bbox'][2][idx], labels['bbox'][3][idx]
        pt1, pt2, pt3, pt4 = int(pt1) , int(pt2), int(pt3), int(pt4)
        print(category)
        img = cv2.imread(img)
        cv2.rectangle(img, (pt1, pt2), [pt1+pt3, pt2+pt4], color=[255, 0, 0], thickness=3)
        cv2.imshow('asdf', img)
        k = cv2.waitKey()
        if k == 120:
            cv2.destroyAllWindows()
            exit()
        idx += 1


if __name__ == '__main__':
    test_dataloader(cfg.DATA)
