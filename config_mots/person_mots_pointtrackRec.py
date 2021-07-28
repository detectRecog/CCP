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
    re_train=True,
    fix_bn=True, # for finetune

    save_dir='./mots_person_pointtrack_rec',
    # resume_path='./mots_person_pointtrack_rec/best-val_acc=78.8600.ckpt_v8_finetune',
    resume_path='./mots_person_pointtrack_rec/best-val_acc=69.2900.ckpt_v8_finetune',
    # checkpoint_path='./mots_person_pointtrack_rec/best-val_acc=79.5600.ckpt', # for test
    val_check_interval=1.0,
    gpus=4,

    train_dataset={
        # 'name': 'mots_cars_CAP',
        'name': 'mots_cars_CAP_v2',
        'kwargs': {
            'root_dir': kittiRoot,
            'class_id': 2,
            'type': 'trainval',
            # 'type': 'train',
            'size': 4000, # for finetune
            # 'size': 10000,
            'check_paste_quality': True,
            'img_size': (384, 1248),
            'add_kins': False, # for finetune
            'transform': my_transforms.get_transform([
                {
                    'name': 'AdjustBrightness', # for finetune
                    'opts': {}
                },
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
        'workers': 6
    },

    val_dataset={
        'name': 'mots_cars_CAP',
        'kwargs': {
            'root_dir': kittiRoot,
            'class_id': 2,
            # 'type': 'test',
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
        'workers': 5,
    },

    model={
        'name': 'pointtrack-v8-rec',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1, 2, 3],
        }
    },

    # lr=1e-4,
    # n_epochs=100,
    lr=5e-5,
    n_epochs=50,
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        # 'foreground_weight': 200,
        'foreground_weight': 500, # for finetune only
        'cls_ratio': 5.0,
    },
    flow_weight=1.0, # for finetune only
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSegClsRecLoss0131Lightn',
    # seg_dir='./person_SE4_test_prediction/',
    seg_dir='./person_kitti_val_prediction_0430/',
    avg_seed=0.7,  # 79.26  91.99  86.38
    threshold=0.93, # 79.26  91.99  86.38
    min_seed_thresh=0.4, # 79.26  91.99  86.38
    inst_ratio=0.8,
    dist_thresh=0.41,
    min_pixel=64,
)


def get_args():
    return copy.deepcopy(args)
