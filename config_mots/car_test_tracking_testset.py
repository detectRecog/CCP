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
    save_dir='./tracks_car_pointtrack_testset/',
    # checkpoint_path='./car_finetune_tracking/best_iou_model.pth85.58_0.002_trainval',
    checkpoint_path='./car_finetune_tracking/best_iou_model.pth91.78_0.0002_M',
    run_eval=True,

    dataset= {
        'name': 'mots_track_val_env_offset',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'val',
            'test': True,
            'num_points': 1500,
            'box': True,
            'gt': False,
            'category': True,
            'ex':0.2
        },
        'batch_size': 1,
        'workers': 32
    },

    model={
        # 'name': 'tracker_offset_emb',
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