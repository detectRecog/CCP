"""
for reproduce
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
    save_dir='./mots_SE4C_person',
    # resume_path='./mots_SE4C_person/checkpoint.pth',
    resume_path='./mots_SE4C_person/best_iou_model.pth78.5800',
    # resume_path='./mots_SE4C_person/best_iou_model.pth77.4900',

    train_dataset = {
        'name': 'mots_insts',
        'kwargs': {
            'class_id': 2,
            'root_dir': kittiRoot,
            'type': 'crop',
            'size': 2400,
            'aug': True,
            'add_person_num': 2,
            'transform': my_transforms.get_transform([
                {
                    'name': 'AdjustBrightness',
                    'opts': {}
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
        # 'batch_size': 8,
        # 'workers': 1,
        'batch_size': 4,
        'workers': 20
    },

    val_dataset = {
        'name': 'mots_persons',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'val',
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
    },

    model={
        'name': 'branched_erfnet_up4',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1, 2],
            'input_channel': 3
        }
    },

    lr=5e-5,
    n_epochs=60,
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        'foreground_weight': 20,
        'cls_ratio': 0.5,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSegClsLoss0904',
    # for eval
    seg_dir='./person_kitti_val_prediction_0430/',
    avg_seed=0.6,
    threshold=0.95,
    min_seed_thresh=0.4,
    inst_ratio=0.8,
    dist_thresh=0.41,
    min_pixel=88,
)


def get_args():
    return copy.deepcopy(args)
