"""
for FlowNet with Fp16
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import torch
import subprocess
from datasets import get_dataset
from models import get_model
from config import *
os.chdir(rootDir)
from config_mots import *
from file_utils import *
from utils.torch_utils import same_seeds
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor


torch.backends.cudnn.enabled = True # Good
# same_seeds()
config_name = sys.argv[1]
args = eval(config_name).get_args()
print('avg_seed: ', args['avg_seed'],
      ', threshold: ', args['threshold'],
      ', min_seed: ', args['min_seed_thresh'],
      ', inst_ratio: ', args['inst_ratio'],
      ', dist_thresh: ', args['dist_thresh'],
      ', min_pixel: ', args['min_pixel'],)

# train dataloader
train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=False, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=False)

# val dataloader
val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=True,
    num_workers=args['val_dataset']['workers'], pin_memory=False)

model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args)

def load_checkpoint(model_path):
    weights = torch.load(model_path, map_location=lambda storage, loc: storage)
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict


if "resume_path" in args.keys() and "re_train" in args.keys():
    ckpt = torch.load(args["resume_path"], map_location=lambda storage, loc: storage)
    # weights = remove_module_in_dict(ckpt['state_dict'])
    # model.load_state_dict(weights, strict=False)
    try:
        model.load_state_dict(ckpt['state_dict'])
    except:
        print('Load weights with strict False')
        model.load_state_dict(ckpt['state_dict'], strict=False)
    print('Load weights successfully for re-train, %s' % args["resume_path"])

# epoch, weights = load_checkpoint('mots_FG_resize_involve/pwclite_ar.tar')
# model.pwc.load_state_dict(weights)
# print('Load PWCLite successfully!')

checkpoint_callback1 = ModelCheckpoint(
    monitor='val_acc',
    dirpath=args['save_dir'],
    filename='best-{val_acc:.4f}',
    save_top_k=3,
    mode='max')
checkpoint_callback2 = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args['save_dir'],
    filename='best-{val_loss:.8f}',
    save_top_k=3,
    mode='min')
checkpoint_callback3 = ModelCheckpoint(
    monitor='seg_iou',
    dirpath=args['save_dir'],
    filename='best-{seg_iou:.4f}',
    save_top_k=2,
    mode='max')
# lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(max_epochs=args['n_epochs'],
                     # val_check_interval=0.1,
                     # distributed_backend="ddp", gpus=4,
                     val_check_interval=1.0 if 'val_check_interval' not in args.keys() else args['val_check_interval'],
                     distributed_backend="ddp", gpus=2 if 'gpus' not in args.keys() else args['gpus'],
                     default_root_dir=args['save_dir'],
                     callbacks=[checkpoint_callback1, checkpoint_callback2, checkpoint_callback3],
                     # amp_level='O2', precision=16,
                     # distributed_backend="dp", gpus=4,
                    precision=32 if 'precision' not in args.keys() else args['precision'],
                     num_sanity_val_steps=2, gradient_clip_val=0.1, sync_batchnorm=True,
                     resume_from_checkpoint=args["resume_path"] if "resume_path" in args.keys() and (not "re_train" in args.keys()) else None)
# ,amp_level='O2', precision=16, num_nodes=4
trainer.fit(model, train_loader, val_loader)

