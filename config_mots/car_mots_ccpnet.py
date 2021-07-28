"""
initialize by car_mots_pointtrack_track
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
    fix_bn=True,

    save_dir='./mots_car_ccpnet',
    # resume_path='./mots_car_ccpnet/best-val_acc=87.3500.ckpt_finetune',
    resume_path='/data/xuzhenbo/PointTrack/mots_car_pointtrack/best-val_acc=92.2800.ckpt_V8_trainval', # testset
    val_check_interval=1.0,
    gpus=4,

    train_dataset={
        # 'name': 'mots_cars_CCAP',
        # 'name': 'mots_cars_CCAP_v2',
        'name': 'mots_cars_CCAP_v3',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'train',
            # 'size': 2000,
            'size': 1000, # finetune
            'img_size': (384, 1248),
            'nearby': 7,
            'non_shift': True,
            # 'randomCropShift': True,
            'border_aug': True,
            'paste_by_order': True,
            'add_kins': False, # finetune
            'max_instance_num': 15, # finetune
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'pad_mask', 'image2', 'instance2', 'label2', 'pad_mask2', 'image3', 'instance3', 'label3', 'pad_mask3'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.LongTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.LongTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 5
    },

    val_dataset={
        'name': 'mots_cars_CCAP',
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
        'name': 'ccpnet_track',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1, 2, 3],
            'n_frames': 3,
        }
    },

    # lr=5e-4,
    # n_epochs=40,
    lr=2e-4, # finetune
    n_epochs=40, # finetune
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        'foreground_weight': 200,
        'cls_ratio': 5.0,
        'max_inst': 15,
    },
    flow_weight=1.0,
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSegClsRecLoss0226Lightn',
    seg_dir='./car_SE_val_prediction_0402/',
    track_dir = './tracks_car_pointtrack_val/',
    avg_seed=0.7,
    threshold=0.9,
    dist_thresh=0.41,
    min_seed_thresh=0.35,
    inst_ratio=0.9,
    min_pixel=128,
)


def get_args():
    return copy.deepcopy(args)
