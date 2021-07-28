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
    save_dir='./tracks_person_pointtrack_val/',
    car=False,
    # checkpoint_path='./mot_finetune_tracking/best_iou_model.pth79.45_0.0002', # MOT
    checkpoint_path='./person_finetune_tracking/best_iou_model.pth75.25_0.0002_M',    # PointTrackV2
    # checkpoint_path='./person_finetune_tracking/best_iou_model.pth62.31_0.002_unfix_near5', # PointTrack++
    # checkpoint_path='./person_finetune_tracking/checkpoint.pth',
    run_eval=True,

    dataset= {
        'name': 'person_track_val_offset',
        'kwargs': {
            'root_dir': kittiRoot,
            'type': 'val',
            'num_points': 1500,
            'box': True,
            'category': True,
            'ex':0.2
        },
        'batch_size': 1,
        'workers': 20
    },

    model={
        'name': 'tracker_offset_emb_randla',
        # 'name': 'tracker_offset_emb',
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
