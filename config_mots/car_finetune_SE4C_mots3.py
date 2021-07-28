"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms
from config import *

n_sigma=2
args = dict(

    cuda=True,
    display=False,
    display_it=5,
    save=True,
    fix_bn=True,

    save_dir='./mots_SE4C_car2',
    # resume_path='./mots_SE4C_car2/checkpoint.pth', # first
    # resume_path='./mots_SE4C_car2/best_focal_model.pth0.00000203_0.3876', # first
    resume_path='./mots_SE4C_car2/best_iou_model.pth89.0700', # 50val

    train_dataset = {
        'name': 'mots_insts',
        'kwargs': {
            'class_id': 1,
            'root_dir': kittiRoot,
            'type': 'train',
            'size': 1600,
            # 'add_kins': True,   # KINS background is different
            'aug': True,
            'transform': my_transforms.get_transform([
                {
                    'name': 'AdjustBrightness',
                    'opts': {}
                },
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        # 'size': (256, 1056),
                        'size': (320, 1024),    # P100
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
                {
                    'name': 'Flip',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                    }
                },
            ]),
        },
        # 'batch_size': 2,
        # 'workers': 1,
        'batch_size': 2,
        'workers': 20
    },

    val_dataset = {
        'name': 'mots_cars',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'val',
            # 'size': 500,
            'transform': my_transforms.get_transform([
                {
                    'name': 'LU_Pad',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'size': (384, 1248),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 2,
        'workers': 20,
        # 'batch_size': 2,
        # 'workers': 6,
    },

    model={
        'name': 'branched_erfnet_up4',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1, 2],
            'input_channel': 3
        }
    },

    # lr=5e-5,  # bad
    # milestones=[10],
    # n_epochs=25,
    lr=5e-6,
    # milestones=[200],
    n_epochs=40,
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        'foreground_weight': 200,
        # 'cls_ratio': 1.0,
        'cls_ratio': 5.0,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSegClsLoss0904',
    seg_dir='./car_SE_val_prediction_0402/',
    avg_seed=0.6,
    threshold=0.9,
    min_seed_thresh=0.4,
    inst_ratio=0.95,
    dist_thresh=0.41,
    min_pixel=128,
)


def get_args():
    return copy.deepcopy(args)
