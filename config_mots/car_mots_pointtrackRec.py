"""
train with CAP
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
    # re_train=True,
    # fix_bn=True, # for finetune

    save_dir='./mots_car_pointtrack_rec',
    resume_path='./mots_car_pointtrack_rec/best-val_acc=87.1100.ckpt_V8_rec',
    # resume_path='./mots_car_pointtrack/best-val_acc=87.0900.ckpt_V8',
    # checkpoint_path='./mots_car_pointtrack/best-val_acc=79.5600.ckpt', # for test
    val_check_interval=1.0,
    gpus=2,

    train_dataset={
        'name': 'mots_cars_CAP',
        # 'name': 'mots_cars_CAP_v3',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'train',
            'size': 4000,
            # 'size': 10000,
            'center_pad': True,
            'img_size': (384, 1248),
            # 'add_kins': False, # for finetune
            # 'add_crop': True, # for raw data
            # 'check_paste_quality': True,
            'transform': my_transforms.get_transform([
                # {
                #     'name': 'AdjustBrightness', # for finetune
                #     'opts': {}
                # },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'pad_mask'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 5
    },

    val_dataset={
        'name': 'mots_cars_CAP',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'val',
            'img_size': (384, 1248),
            # 'centerRadius': True,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'pad_mask'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 3,
    },

    model={
        'name': 'pointtrack-v8-rec',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1, 2, 3],
        }
    },

    # lr=1e-4,
    # n_epochs=40,
    lr=5e-5,
    n_epochs=40,
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        'foreground_weight': 200, # for finetune only
        'cls_ratio': 5.0, # for finetune only
        # 'foreground_weight': 20,
        # 'cls_ratio': 1.0,
    },
    flow_weight=1.0, # for finetune only
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSegClsRecLoss0131Lightn',
    seg_dir='./car_SE_val_prediction_0402/',
    avg_seed=0.7,
    threshold=0.9,
    dist_thresh=0.41,
    min_seed_thresh=0.35,
    inst_ratio=0.9,
    min_pixel=128,
)


def get_args():
    return copy.deepcopy(args)
