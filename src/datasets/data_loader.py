import torch
import numpy as np
from PIL import Image
from torch.ao.nn.quantized.functional import upsample
from torch.utils import data
import torchvision.transforms as transforms
import torchvision as tv
import torchvision.transforms.functional as TF
import os

from . import memory_dataset as memd
from .dataset_config import dataset_config
import cv2

def get_all_file(path):
    return [os.path.abspath(os.path.join(path, p)) for p in sorted(os.listdir(path))]


def get_loaders(dataset, batch_sz, num_work, pin_mem, validation=.1):
    """Apply transformations to Dataset and create the DataLoaders for each task"""

    # get configuration for current datasets
    dc = dataset_config[dataset]

    # transformations
    trn_transform, tst_transform = get_transforms(resize=dc['resize'], pad=dc['pad'], crop=dc['crop'],
                                                  flip=dc['flip'], normalize=dc['normalize'],
                                                  extend_channel=dc['extend_channel'],
                                                  elastic=dc['elastic'], color_jitter=dc['color_jitter'],
                                                  blur=dc['blur'])

    # datasets
    trn_dset, val_dset, tst_dset = get_dataset(dataset, dc['path'], validation=validation,
                                               trn_transform=trn_transform, tst_transform=tst_transform)

    # loaders
    trn_load = data.DataLoader(trn_dset, batch_size=batch_sz, shuffle=True, num_workers=num_work, pin_memory=pin_mem)
    val_load = data.DataLoader(val_dset, batch_size=batch_sz, shuffle=False, num_workers=num_work, pin_memory=pin_mem)
    tst_load = data.DataLoader(tst_dset, batch_size=batch_sz, shuffle=False, num_workers=num_work, pin_memory=pin_mem)
    return trn_load, val_load, tst_load


def get_dataset(dataset, path, validation, trn_transform, tst_transform):
    """Extract datasets and create Dataset class"""

    if dataset == 'cifar10':
        trn_data_cifar = tv.datasets.CIFAR10(os.path.join(path, 'train'), download=True, train=True)
        tst_data_cifar = tv.datasets.CIFAR10(os.path.join(path, 'test'), download=True, train=False)
        trn_data = {'x': trn_data_cifar.data, 'y': trn_data_cifar.targets}
        tst_data = {'x': tst_data_cifar.data, 'y': tst_data_cifar.targets}
        all_data = memd.get_data(trn_data, tst_data, validation=validation)

    elif dataset == 'cifar10_debug':
        trn_data_cifar = tv.datasets.CIFAR10(os.path.join(path, 'train'), download=True, train=True)
        tst_data_cifar = tv.datasets.CIFAR10(os.path.join(path, 'test'), download=True, train=False)
        trn_data = {'x': trn_data_cifar.data[:1000], 'y': trn_data_cifar.targets[:1000]}
        tst_data = {'x': tst_data_cifar.data[:1000], 'y': tst_data_cifar.targets[:1000]}
        all_data = memd.get_data(trn_data, tst_data, validation=validation)

    else:
        # Other datasets with our format -- path needs to have a train.txt and a test.txt with image-label pairs
        trn_data, tst_data = {'x': [], 'y': []}, {'x': [], 'y': []}
        # read filenames and labels
        trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)
        # parse them into required structure
        for this_image, this_label in trn_lines:
            trn_data['x'].append(this_image)
            trn_data['y'].append(this_label)
        for this_image, this_label in tst_lines:
            tst_data['x'].append(this_image)
            tst_data['y'].append(this_label)
        # compute splits
        all_data = memd.get_data(trn_data, tst_data, validation=validation)

    # wrap datasets
    trn_dset = memd.MemoryDataset(all_data['trn'], trn_transform)
    val_dset = memd.MemoryDataset(all_data['val'], trn_transform)
    tst_dset = memd.MemoryDataset(all_data['tst'], tst_transform)

    return trn_dset, val_dset, tst_dset


def get_transforms(resize, pad, crop, flip, normalize, extend_channel, elastic, color_jitter, blur):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = [transforms.ToTensor(), transforms.Resize((224,224))]
    tst_transform_list = [transforms.ToTensor(), transforms.Resize((224,224))]


    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize, antialias=None))
        tst_transform_list.append(transforms.Resize(resize, antialias=None))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop, antialias=None))
        # tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())
        trn_transform_list.append(transforms.RandomVerticalFlip())
        trn_transform_list.append(transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5))

    # if color_jitter:
    #     img_transform_list.append(transforms.ColorJitter())
    #
    # if blur is not None:
    #     img_transform_list.append(transforms.GaussianBlur(kernel_size=blur))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), transforms.Compose(tst_transform_list)
