'''
自上而下，窄宽窄，1/4,1/8,1/16融合
'''

import torch
import torch.nn as nn
import cv2
import subprocess
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.utils import ClusterSeedClsWithFilter
from criterions.my_loss import *
from file_utils import *
from utils.mots_util import plot_instanceMap
from config import *
from PIL import Image
from copy import copy


class PointTrackLightning(pl.LightningModule):
    def __init__(self, num_classes, mot17=False, c1=64, c2=256, c3=256, c4=64, oc2=64):
        # by default, c1=64, c2=256, c3=256, c4=64
        super().__init__()
        self.args, self.criterion, self.milestones = None, None, None
        print('Creating PointTrackV8Rec with {} classes'.format(num_classes))

        self.num_classes = num_classes

        from models.erfnetGroupNorm import Encoder0126, Decoder0118
        self.backbone = Encoder0126(c1=c1, c2=c2, c3=c3, c4=c4)
        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(Decoder0118(n, ic=c2+c3+c4, c2=oc2))

        self.cluster = ClusterSeedClsWithFilter(larger=mot17)
        # self.cluster = ClusterSeedClsWithBBoxs()
        self.val_loss, self.val_acc = 100, 0
        self.flow_weight = 1.0
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_output(self, args):
        n_sigma = args['loss_opts']['n_sigma']
        self.args = args
        if 'milestones' in self.args.keys():
            self.milestones = args['milestones']
        self.criterion = eval(self.args['loss_type'])(**self.args['loss_opts'])
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ', output_conv.weight.size())
            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)
            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)
        # self.load_from('./mots_person_pointtrack_rec/best-val_acc=68.5400.ckpt_v8_train')

    def load_from(self, path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']
        ckpt = remove_key_word(ckpt, ['criterion.xym'])
        self.load_state_dict(ckpt, strict=False)
        print('resume checkpoint with strict False from %s' % path)

    def configure_optimizers(self):
        params = list(self.named_parameters())

        def is_backbone(n): return '.backbone' in n
        def is_bn(n): return '.bn' in n

        if 'fix_bn' in self.args.keys() and self.args['fix_bn']:
            grouped_parameters = [
                {"params": [p for n, p in params if (not is_bn(n)) and (not is_backbone(n))], 'lr': self.args['lr']},
                {"params": [p for n, p in params if (not is_bn(n)) and (is_backbone(n))], 'lr': self.args['lr'] * 0.1}
            ]
            print('BN fixed')
        else:
            grouped_parameters = [
                {"params": [p for n, p in params if not is_backbone(n)], 'lr': self.args['lr']},
                {"params": [p for n, p in params if is_backbone(n)], 'lr': self.args['lr']*0.1}
            ]
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.args['lr'])
        print('Backbone LR *0.1!')
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.args['lr'], weight_decay=1e-4)

        if self.milestones is None:
            def lambda_(epoch):
                return pow((1 - ((epoch) / self.args['n_epochs'])), 0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, gamma=0.4)
        return [optimizer], [scheduler]

    def forward(self, ims):
        output2, output3, output4 = self.backbone(ims)
        output = torch.cat([output2, F.interpolate(output3, scale_factor=(2.0, 2.0), mode='nearest'), F.interpolate(output4, scale_factor=(4.0, 4.0), mode='nearest')], dim=1)
        output = torch.cat([decoder.forward(output) for decoder in self.decoders], 1)
        return output

    def training_step(self, sample, batch_nb):
        # REQUIRED
        ims = sample['image']
        instances = sample['instance'].squeeze(1)
        class_labels = sample['label'].squeeze(1)
        output = self(ims)

        seed_w = sample['seed_w'][0] if 'seed_w' in sample.keys() else None
        loss, focal_loss, _ = self.criterion(output, ims, instances, class_labels, **self.args['loss_w'], seed_w=seed_w)
        flow_loss = focal_loss
        return {'loss': loss, 'focal_loss': focal_loss, 'flow_loss': flow_loss}

    def training_step_end(self, batch_parts):
        return {'loss': batch_parts['loss'].mean(), 'focal_loss': batch_parts['focal_loss'].mean(), 'flow_loss': batch_parts['flow_loss'].mean()}

    def training_epoch_end(self, train_step_outputs):
        if self.global_rank == 0:
            train_loss, focal_loss, flow_loss = 0, 0, 0
            for v in train_step_outputs:
                train_loss += v['loss'].item()
                focal_loss += v['focal_loss'].item()
                flow_loss += v['flow_loss'].item()
            train_loss = train_loss / len(train_step_outputs)
            focal_loss = focal_loss / len(train_step_outputs)
            flow_loss = flow_loss / len(train_step_outputs)
            print('')
            print('===> train loss: {:.8f}, ===> focal_loss: {:.8f}, ===> flow_loss: {:.8f}'.format(train_loss, focal_loss, flow_loss))
            print('Learning Rate {:.8f}'.format(self.trainer.optimizers[0].param_groups[0]['lr']))

    def validation_epoch_start(self):
        self.eval()
        torch.cuda.empty_cache()

    def validation_step(self, sample, batch_idx):
        with torch.no_grad():
            ims = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            output = self(ims)
            loss, focal_loss, ious = self.criterion(output, ims, instances, class_labels, **self.args['loss_w'], iou=True)

            sizes = sample['im_shape'].squeeze(1)
            im_names = sample['im_name']
            args = self.args
            instance_maps = self.cluster(output, threshold=args['threshold'], min_pixel=args['min_pixel'],
                                       min_inst_pixel=args['min_inst_pixel'] if "min_inst_pixel" in args.keys() else args['min_pixel'],
                                       min_seed_thresh=args['min_seed_thresh'] if "min_seed_thresh" in args.keys() else 0.5,
                                       dist_thresh=args['dist_thresh'] if "dist_thresh" in args.keys() else 0.5,
                                       inst_ratio=args['inst_ratio'] if "inst_ratio" in args.keys() else 0.5,
                                       n_sigma=args["n_sigma"] if "n_sigma" in args.keys() else 2,
                                       avg_seed=args["avg_seed"] if "avg_seed" in args.keys() else 0.0)
            for ind in range(instance_maps.shape[0]):
                (w, h) = sizes[ind]
                video, frameCount = im_names[ind].split('/')[-2:]
                frameCount = int(float(frameCount.split('.')[0]))
                base = video + '_' + str(frameCount) + '.pkl'
                instance_map_np = instance_maps[ind].cpu().numpy()
                instance_map_np = instance_map_np[:h.item(), :w.item()]
                if 'mot17' in self.args['save_dir'] and w.item()==1280:
                    instance_map_np = cv2.resize(instance_map_np, None, fx=0.5, fy=1/1.5, interpolation=cv2.INTER_NEAREST)
                    assert instance_map_np.shape == (480, 640)
                # if 'mot17' in self.args['save_dir'] and w.item()==960:
                #     instance_map_np = cv2.resize(instance_map_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
                #     assert instance_map_np.shape == (1080, 1920)
                save_pickle2(os.path.join(args['seg_dir'], base), instance_map_np)
            return {'val_loss': loss, 'focal_loss': focal_loss, 'iou_sum': sum(ious), 'iou_count': len(ious)}

    def validation_step_end(self, batch_parts):
        # do something with both outputs
        iou_sum = sum(batch_parts['iou_sum']) if isinstance(batch_parts['iou_sum'], list) else batch_parts['iou_sum']
        iou_count = sum(batch_parts['iou_count']) if isinstance(batch_parts['iou_count'], list) else batch_parts['iou_count']
        if iou_count>0 and iou_sum / float(iou_count) < 0.7 and iou_count > 5:
            self.validation_epoch_end([])
        return {
            'val_loss': batch_parts['val_loss'].mean(),
            'focal_loss': batch_parts['focal_loss'].mean(),
            'iou_sum': sum(batch_parts['iou_sum']) if isinstance(batch_parts['iou_sum'], list) else batch_parts['iou_sum'],
            'iou_count': sum(batch_parts['iou_count']) if isinstance(batch_parts['iou_count'], list) else batch_parts['iou_count']
        }

    def validation_epoch_end(self, validation_step_outputs):
        if len(validation_step_outputs) < 10: # sanity check
            return
        val_loss, focal_loss, iou_sum, iou_count = 0,0,0,0
        for v in validation_step_outputs:
            val_loss += v['val_loss'].item()
            focal_loss += v['focal_loss'].item()
            iou_sum += v['iou_sum']
            iou_count += v['iou_count']
        val_loss = val_loss/len(validation_step_outputs)
        focal_loss = focal_loss/len(validation_step_outputs)
        val_iou = iou_sum / iou_count

        self.eval()
        torch.cuda.empty_cache()
        if self.global_rank == 0:
            if 'mot17' in self.args['save_dir']:
                p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'mot_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
            elif 'apollo' in self.args['save_dir']:
                p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'apollo_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
            elif 'person' in self.args['save_dir']:
                p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'person_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
            else:
                p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'car_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)

            pout = p.stdout.decode("utf-8")
            if 'person' in self.args['save_dir']:
                class_str = "Evaluate class: Pedestrians"
            else:
                class_str = "Evaluate class: Cars"
            pout = pout[pout.find(class_str):]
            print(pout[pout.find('all   '):][6:126].strip())
            acc_str = pout[pout.find('all   '):][6:26].strip().split(' ')[0]
            if len(acc_str) > 0:
                val_acc = float(acc_str)
            else:
                val_acc = 0.0
            print('####### val_acc: {:.4f}, seg iou: {:.4f}, val seed: {:.8f}, val loss: {:.8f}'.format(val_acc, val_iou, focal_loss, val_loss))
            self.log('seg_iou', val_iou)
            self.log('val_acc', val_acc)
            self.log('val_loss', val_loss)
        else:
            val_acc, val_iou = 0.0, 0.0

    def test_step(self, sample, batch_idx):
        with torch.no_grad():
            ims = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            loss, focal_loss, ious = torch.zeros([1]).type_as(ims), torch.zeros([1]).type_as(ims), torch.zeros([1]).type_as(ims)
            im_names = sample['im_name']
            video, frameCount = im_names[0].split('/')[-2:]
            frameCount = int(float(frameCount.split('.')[0]))
            output = self(ims)
            sizes = sample['im_shape'].squeeze(1)

            args = copy(self.args)
            instance_maps = self.cluster(output, threshold=args['threshold'], min_pixel=args['min_pixel'],
                                         min_inst_pixel=args['min_inst_pixel'] if "min_inst_pixel" in args.keys() else
                                         args['min_pixel'],
                                         min_seed_thresh=args['min_seed_thresh'] if "min_seed_thresh" in args.keys() else 0.5,
                                         dist_thresh=args['dist_thresh'] if "dist_thresh" in args.keys() else 0.5,
                                         inst_ratio=args['inst_ratio'] if "inst_ratio" in args.keys() else 0.5,
                                         n_sigma=args["n_sigma"] if "n_sigma" in args.keys() else 2,
                                         avg_seed=args["avg_seed"] if "avg_seed" in args.keys() else 0.0)

            for ind in range(instance_maps.shape[0]):
                (w, h) = sizes[ind]
                video, frameCount = im_names[ind].split('/')[-2:]
                frameCount = int(float(frameCount.split('.')[0]))
                base = video + '_' + str(frameCount) + '.pkl'
                instance_map_np = instance_maps[ind].cpu().numpy()
                instance_map_np = instance_map_np[:h.item(), :w.item()]
                if 'mot17' in self.args['save_dir'] and w.item()==1280:
                    instance_map_np = cv2.resize(instance_map_np, None, fx=0.5, fy=1/2.0, interpolation=cv2.INTER_NEAREST)
                    try:
                        assert instance_map_np.shape == (480, 640)
                    except:
                        print(instance_map_np.shape)
                        exit(0)
                save_pickle2(os.path.join(args['seg_dir'], base), instance_map_np)
            return {'val_loss': loss, 'focal_loss': focal_loss, 'iou_sum': sum(ious), 'iou_count': len(ious)}

    def test_step_end(self, batch_parts):
        # do something with both outputs
        return {
            'val_loss': batch_parts['val_loss'].mean(),
            'focal_loss': batch_parts['focal_loss'].mean(),
            'iou_sum': sum(batch_parts['iou_sum']) if isinstance(batch_parts['iou_sum'], list) else batch_parts['iou_sum'],
            'iou_count': sum(batch_parts['iou_count']) if isinstance(batch_parts['iou_count'], list) else batch_parts['iou_count']
        }

    def test_epoch_end(self, validation_step_outputs):
        val_loss, focal_loss, iou_sum, iou_count = 0,0,0,0
        for v in validation_step_outputs:
            val_loss += v['val_loss'].item()
            focal_loss += v['focal_loss'].item()
            iou_sum += v['iou_sum']
            iou_count += v['iou_count']
        val_loss = val_loss/len(validation_step_outputs)
        focal_loss = focal_loss/len(validation_step_outputs)
        val_iou = iou_sum / iou_count

        self.eval()
        torch.cuda.empty_cache()
        if self.global_rank == 0:
            if 'mot17' in self.args['save_dir']:
                # p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'mot_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
                p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'mot_test_tracking_testset'], stdout=subprocess.PIPE, cwd=rootDir)
            elif 'person' in self.args['save_dir']:
                # p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu_new.py", 'person_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
                # p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu_new.py", 'person_test_tracking_testset'], stdout=subprocess.PIPE, cwd=rootDir)
                p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'person_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
                # p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'person_test_tracking_testset'], stdout=subprocess.PIPE, cwd=rootDir)
            else:
                p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu.py", 'car_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
                # p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu_new.py", 'car_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
                # p = subprocess.run([pythonPath, "-u", "test_tracking_one_gpu_new.py", 'car_test_tracking_testset'], stdout=subprocess.PIPE, cwd=rootDir)

            if not 'mot17' in self.args['save_dir']:
                pout = p.stdout.decode("utf-8")
                if 'person' in self.args['save_dir']:
                    class_str = "pedestrian"
                else:
                    class_str = "car"
                pout = pout[pout.find(class_str):]
                print(pout)
                acc_str = '0.0'
            else:
                pout = p.stdout.decode("utf-8")
                if 'person' in self.args['save_dir']:
                    class_str = "Evaluate class: Pedestrians"
                else:
                    class_str = "Evaluate class: Cars"
                pout = pout[pout.find(class_str):]
                print(pout[pout.find('all   '):][6:126].strip())
                acc_str = pout[pout.find('all   '):][6:26].strip().split(' ')[0]
            if len(acc_str) > 0:
                val_acc = float(acc_str)
            else:
                val_acc = 0.0
            print('####### val_acc: {:.4f}, seg iou: {:.4f}, val seed: {:.8f}, val loss: {:.8f}'.format(val_acc, val_iou, focal_loss, val_loss))
            self.log('seg_iou', val_iou)
            self.log('val_acc', val_acc)
            self.log('val_loss', val_loss)

