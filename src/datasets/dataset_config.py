from os.path import join

_BASE_DATA_PATH = "./data/"

dataset_config = {
    'monuseg': {
        'path': join(_BASE_DATA_PATH, 'MoNuSeg'),
        'resize': None,
        'pad': None,
        'crop': 572,
        'flip': True,
        'color_jitter': True,
        'blur': 5,
        # 'normalize': ((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
        # 'normalize': ((0.6040, 0.4475, 0.6439), (0.1877, 0.2324, 0.2309)),
        # 'normalize': ((0.6610, 0.4633, 0.5951), (0.2052, 0.2126, 0.1766)), # mit ColorNorm
        # 'normalize': ((0.5286, 0.3805, 0.4706), (0.2841, 0.2529, 0.2560)), # mit ColorNorm und adapthist
    },
    'cifar10': {
        'path': join(_BASE_DATA_PATH, 'CIFAR-10'),
        'normalize': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        'resize': None,
        'pad': None,
    },
    'cifar10_debug': {
        'path': join(_BASE_DATA_PATH, 'CIFAR-10'),
        'normalize': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        'resize': None,
        'pad': None,
    }

}

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel', 'elastic', 'color_jitter', 'blur']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
