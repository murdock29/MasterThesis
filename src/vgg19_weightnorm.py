import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torchvision.models import vgg19_bn#
import os
from glob import glob
import matplotlib.pyplot as plt






def compute_covariance_matrix(features):
    # Center the data
    # features_centered = features - torch.mean(features, dim=0, keepdim=True)
    # # Compute the covariance matrix
    # cov_matrix = torch.mm(features_centered.T, features_centered) / (features.size(0) - 1)
    cov_matrix = torch.cov(features.T)

    return cov_matrix

def compute_singular_values(cov_matrix):
    """
    Compute singular values using SVD with PyTorch.
    """
    # Perform SVD on the GPU
    _, singular_values, _ = torch.svd(cov_matrix)
    return singular_values

def compute_numerical_rank(singular_values, threshold=1e-3):
    """
    Compute the numerical rank based on singular value thresholding.
    """
    max_singular_value = torch.max(singular_values)
    # Identify significant singular values
    significant_singular_values = singular_values[singular_values > threshold * max_singular_value]
    numerical_rank = significant_singular_values.size(0)
    return numerical_rank

import time
def calculate_numerical_rank(features, threshold=1e-3):
    # Compute sample covariance matrix
    start = time.time()
    sample_cov_matrix = compute_covariance_matrix(features)
    first_time = time.time() - start
    # Compute singular values
    singular_values = compute_singular_values(sample_cov_matrix)
    second_time = time.time() - start - first_time
    # Compute numerical rank
    numerical_rank = compute_numerical_rank(singular_values, threshold)
    third_time = time.time() - start - first_time - second_time
    print(f"First Time: {first_time}, Second Time: {second_time}, Third Time: {third_time}")
    return numerical_rank, singular_values

def calculate_norm(model, model_next):
    norms = []
    i = 0
    next_params = model_next.named_parameters()
    for name, param in model.named_parameters():
        next_name, next_param = next(next_params)
        if 'weight' in name:
            if int(name.split('.')[1]) in conv2d_indices:
                norm = torch.norm(param - next_param, p=2).item() / np.sqrt(param.numel())
                norms.append(norm)
                i += 1
    return norms



if __name__ == '__main__':
    model = vgg19_bn()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    next_model = vgg19_bn()
    next_model.classifier[-1] = nn.Linear(next_model.classifier[-1].in_features, 10)
    model.load_state_dict(torch.load('../model/3_2_OOD/vgg19_checkpoint_epoch_010.pt'))
    l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    checkpoint_files = sorted(glob('../model/3_2_OOD/vgg19_checkpoint_epoch_*.pt'))
    weight_norms = {name: [] for name, _ in model.named_parameters() if 'weight' in name}

    conv2d_indices = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49] #16 conv-layers
    classifier_indices = [51, 52] # linear_layers without head
    layer_indices = conv2d_indices + classifier_indices
    checkpoint = torch.load(checkpoint_files[0])
    model.load_state_dict(checkpoint)

    # next_checkpoint = torch.load(checkpoint_files[1])
    # next_model.load_state_dict(next_checkpoint)

    # norms = calculate_norm(model, next_model)
    row_list = []
    for file in checkpoint_files[1:]:

        next_checkpoint = torch.load(file)
        next_model.load_state_dict(next_checkpoint)
        norms = calculate_norm(model, next_model)
        row_list.append(norms)
        model = copy.deepcopy(next_model)
        # checkpoints = next_checkpoint


    matrix = np.array(row_list)
    matrix = np.clip(matrix, 0, 0.02) #todo clip for different scales
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)

    # Set the ticks and labels for x and y axes
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))

    # Label the axes
    ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1))
    ax.vlines(7, 0, matrix.shape[0]-1, linestyles='dashed', colors='r')
    ax.set_xlabel('layers')
    ax.set_ylabel('Checkpoints to each other')

    # Add a title
    ax.set_title('Weightnorm matrix')

    # Show the plot
    plt.show()

    print('end')