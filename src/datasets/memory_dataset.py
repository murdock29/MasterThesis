import torch
import torchvision
from torch.utils.data import Dataset
import random
import numpy as np
from sklearn.model_selection import train_test_split
import kornia
import cv2
import torch.nn.functional as F

class MemoryDataset(Dataset):
    """Characterizes a datasets for PyTorch -- this datasets pre-loads all images in memory"""

    def __init__(self, data, transform):
        """Initialization"""
        self.masks = data['y']
        self.images = data['x']
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index].astype(np.float32) / 255.0
        x = self.transform(x)
        # x = x.transpose(2, 0, 1)
        y = torch.tensor(self.masks[index])
        y_one_hot = F.one_hot(y, num_classes=10).float()
        return x, y_one_hot  # we need to add the extra dimension in front again


def get_data(trn_data, tst_data, validation):
    """Prepare data: datasets splits, task partition, class order"""

    # initialize data structure
    data = {'trn': {'x': [], 'y': []}, 'val': {'x': [], 'y': []}, 'tst': {'x': [], 'y': []}}
    data['trn'] = trn_data
    data['tst'] = tst_data

    # validation
    if validation > 0.0:
        # raise Exception('Validation not implemented yet')
        # images = 30
        # rnd_img = random.sample(range(images), int(np.round(images * validation)))
        # rnd_img_idx = [idx for idx, fname in enumerate(trn_data['f']) if int(fname.split('/')[-1].split('.')[0].split('_')[1]) in rnd_img]
        # rnd_img_idx.sort(reverse=True)
        # for ii in rnd_img_idx:
        #     data['val']['x'].append(trn_data['x'][ii])
        #     data['val']['y'].append(trn_data['y'][ii])
        #     data['trn']['x'].pop(ii)
        #     data['trn']['y'].pop(ii)
        rnd_img = random.sample(range(len(data['trn']['x'])), int(np.round(len(data['trn']['x']) * validation)))
        rnd_img.sort(reverse=True)
        for ii in range(len(rnd_img)):
            data['val']['x'].append(data['trn']['x'][rnd_img[ii]])
            data['val']['y'].append(data['trn']['y'][rnd_img[ii]])
            data['trn']['x'].pop(rnd_img[ii])
            data['trn']['y'].pop(rnd_img[ii])
    return data
