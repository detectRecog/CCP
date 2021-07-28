"""
initialize by person_mots_pointtrack_track
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

    save_dir='./mots_person_ccpnet',
    # resume_path='./mots_person_ccpnet/best-val_acc=69.35.ckpt_finetune',
    resume_path='/data/xuzhenbo/PointTrack/mots_person_pointtrack_rec/best-val_acc=80.4800.ckpt_rec', # testset
    val_check_interval=1.0,
    gpus=4,

    train_dataset={
        # 'name': 'mots_cars_CCAP',
        # 'name': 'mots_cars_CCAP_v2',
        'name': 'mots_cars_CCAP_v3',
        'kwargs': {
            'root_dir': kittiRoot,
            'class_id': 2,
            'type': 'train',
            'size': 1500,
            'img_size': (384, 1248),
            'nearby': 5,
            'non_shift': True,
            'border_aug': True,
            'randomCropShift': True,
            'paste_by_order': True,
            'add_kins': False, # finetune
            'max_instance_num': 25, # finetune
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
            'class_id': 2,
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
        'name': 'ccpnet_track',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1, 2, 3],
            'n_frames': 2,
        }
    },

    lr=5e-4,
    n_epochs=30,
    # lr=5e-5,
    # n_epochs=50,
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
    flow_weight=1.0, # for finetune only
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSegClsRecLoss0226Lightn',
    seg_dir='./person_kitti_val_prediction_0430/',
    track_dir = './tracks_person_pointtrack_val/',
    avg_seed=0.72,
    threshold=0.938,
    min_seed_thresh=0.4,
    inst_ratio=0.9,
    dist_thresh=0.41,
    min_pixel=120,
)


def get_args():
    return copy.deepcopy(args)
