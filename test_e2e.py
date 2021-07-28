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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
same_seeds()
config_name = sys.argv[1]
args = eval(config_name).get_args()

# val dataloader
val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=False,
    num_workers=args['val_dataset']['workers'], pin_memory=False)

model = get_model(args['model']['name'], args['model']['kwargs'])

# ts = [0.71,0.713,0.718,0.723,0.728,0.73]
# ts = [0.933,0.935]
# ts = [0.38,0.39,0.4,0.42]
# ts = [0.89,0.91,0.92]
# ts = [0.34,0.36]
# ts = [128,136,144]
# ts = [96,128,144,160,176,192,208]

# ts = [0.3,0.5,0.7, 0.8,0.82,0.85] # 43
# ts = [0.1]
ts = [1]
import time
# start = time.time()
with torch.no_grad():
    for t in ts:
        print(t)
        # args['last_discount'] = t
        # args['avg_seed'] = t
        # args['threshold'] = t
        # args['dist_thresh'] = t
        # args['inst_ratio'] = t
        # args['min_seed_thresh'] = t
        # args['min_pixel'] = t

        print('avg_seed', args['avg_seed'],
              'threshold', args['threshold'],
              'min_seed_thresh', args['min_seed_thresh'],
              'inst_ratio', args['inst_ratio'],
              'dist_thresh', args['dist_thresh'],
              'min_pixel', args['min_pixel'])
        model.init_output(args)
        if "resume_path" in args.keys():
            ckpt = torch.load(args["resume_path"], map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['state_dict'])
            print('Load weights successfully, %s' % args["resume_path"])
        trainer = pl.Trainer(max_epochs=args['n_epochs'],
                             # val_check_interval=1.0 if 'val_check_interval' not in args.keys() else args['val_check_interval'],
                             distributed_backend="ddp", gpus=2 if 'gpus' not in args.keys() else args['gpus'],
                             default_root_dir=args['save_dir'],
                             num_sanity_val_steps=0, precision=32 if 'precision' not in args.keys() else args['precision'],)
        trainer.test(model, val_loader, args["resume_path"], verbose=True)

# print(time.time()-start)
'''
mv 0012.txt MOTS20-12.txt
mv 0007.txt MOTS20-07.txt
mv 0006.txt MOTS20-06.txt
mv 0001.txt MOTS20-01.txt
'''