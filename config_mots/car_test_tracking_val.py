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

args = dict(

    cuda=True,
    display=False,

    save=True,
    save_dir='./tracks_car_pointtrack_val/',
    # checkpoint_path='./car_finetune_tracking/checkpoint.pth',
    # checkpoint_path='./apollo_finetune_tracking/best_iou_model.pth72.25_0.0002_M_apollo', # apollo
    # checkpoint_path='./car_finetune_tracking/best_iou_model.pth85.56_0.002_unfix_near5_trainset', # pointTrack++
    checkpoint_path='./car_finetune_tracking/best_iou_model.pth91.78_0.0002_M', # randla
    # checkpoint_path='./pointTrack_weights/PointTrack.pthCar',
    run_eval=True,

    dataset= {
        'name': 'mots_track_val_env_offset',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'val',
            'test': False,
            'num_points': 1500,
            'box': True,
            'gt': False,
            'category': True,
            'ex':0.2
        },
        'batch_size': 1,
        'workers': 4
    },

    model={
        'name': 'tracker_offset_emb_randla',
        'kwargs': {
            'num_points': 1000,
            'margin': 0.2,
            'border_ic': 3,
            'env_points': 500,
            'outputD': 32,
            'category': True,
            # 'POS': True,
            # 'ENV': True,
            # 'FG': True,
        }
    },
    max_disparity=192.0,
    with_uv=True
)


def get_args():
    return copy.deepcopy(args)
