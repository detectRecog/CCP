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

    save_dir='./mots_SE4C_person',
    # resume_path='./mots_SE4C_person/checkpoint.pth',  # crop model
    # resume_path='./mots_SE4C_person/best_iou_model.pth77.5800',  # crop model
    resume_path='./mots_SE4C_person/best_iou_model.pth78.5800',  # crop model

    train_dataset = {
        'name': 'mots_insts',
        'kwargs': {
            'class_id': 2,
            'root_dir': kittiRoot,
            'type': 'train',
            'size': 1600,
            # 'aug': True,
            # 'add_person_num': 2,
            'transform': my_transforms.get_transform([
                {
                    'name': 'AdjustBrightness',
                    'opts': {}
                },
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'size': (320, 832),    # last finetune on P100,
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
        'batch_size': 4,
        'workers': 32,
        # 'batch_size': 2,
        # 'workers': 1
    },

    val_dataset = {
        'name': 'mots_persons',
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
        'batch_size': 4, # the number of GPU
        'workers': 32,
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
    # milestones=[6],
    # n_epochs=100,
    # lr=5e-6,
    # n_epochs=30,
    lr=2e-6,
    n_epochs=30,
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        'foreground_weight': 200,
        'cls_ratio': 5.0,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSegClsLoss0904',
    # for eval
    seg_dir='./person_kitti_val_prediction_0430/',
    avg_seed=0.7,
    threshold=0.95,
    min_seed_thresh=0.4,
    inst_ratio=0.8,
    dist_thresh=0.41,
    min_pixel=88,
)


def get_args():
    return copy.deepcopy(args)
