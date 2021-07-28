"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import torch.nn as nn
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torch
from config import *


class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)


class Visualizer:

    def __init__(self, keys):
        self.wins = {k:None for k in keys}

    def display(self, image, key):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1
    
        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)
    
        fig, ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1
    
        assert n_images == n_axes
    
        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image))
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]))
    
        plt.draw()
        self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


class ClusterSeedClsOld(nn.Module):
    def __init__(self, dim=2):
        super().__init__()

        if dim ==2:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:self.dim]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[self.dim + 2 + n_sigma:n_sigma + self.dim + 3])  # 1 x h x w

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        return instance_map.unsqueeze(0)


class ClusterSeedCls(nn.Module):
    def __init__(self, dim=2):
        super().__init__()

        if dim ==2:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:self.dim]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        # seed_map = prediction[self.dim+1 + n_sigma:n_sigma + self.dim+3]  # 1 x h x w
        # seed_map = F.softmax(seed_map, dim=0)[1:]
        seed_map = torch.sigmoid(prediction[self.dim+1 + n_sigma:n_sigma + self.dim+3])
        seed_map = seed_map[1:]

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel) and count < 200:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        # check inst pixel < min_inst_pixel
        ids = instance_map.unique()[1:]
        for id in ids:
            # print((instance_map==id).sum().item())
            if (instance_map==id).sum() < min_inst_pixel:
                instance_map[instance_map==id] = 0
        return instance_map.unsqueeze(0)


class ClusterSeedClsOffsetShift(nn.Module):
    def __init__(self, dim=2):
        super().__init__()

        if dim ==2:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        offset = torch.tanh(prediction[0:self.dim])
        spatial_emb = offset + xym_s.type_as(prediction)  # 2 x h x w
        h, w = spatial_emb.shape[1:]

        grid = spatial_emb.clone()
        grid[0] = 2.0 * (grid[0] * 1024 / max(w - 1, 1) - 0.5)
        grid[1] = 2.0 * (grid[1] * 1024 / max(h - 1, 1) - 0.5)
        grid = grid.permute(1,2,0)
        offsetSampled = F.grid_sample(offset.unsqueeze(0), grid.unsqueeze(0)).squeeze(0)
        spatial_emb = spatial_emb + offsetSampled

        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        seed_map = prediction[self.dim+1 + n_sigma:n_sigma + self.dim+3]  # 1 x h x w
        seed_map = F.softmax(seed_map, dim=0)[1:]

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        # check inst pixel < min_inst_pixel
        ids = instance_map.unique()[1:]
        for id in ids:
            # print((instance_map==id).sum().item())
            if (instance_map==id).sum() < min_inst_pixel:
                instance_map[instance_map==id] = 0
        return instance_map.unsqueeze(0)


class ClusterSeedClsSimple(nn.Module):
    def __init__(self, dim=2):
        super().__init__()

        if dim ==2:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:self.dim]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        seed_map = prediction[self.dim+1 + n_sigma:n_sigma + self.dim+3]  # 1 x h x w
        seed_map = F.softmax(seed_map, dim=0)[1:]

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        return instance_map.unsqueeze(0)


class ClusterSeedClsPlus(nn.Module):
    def __init__(self, dim=2):
        super().__init__()

        if dim ==2:
            xm = torch.linspace(0, 3, 3072).view(1, 1, -1).expand(1, 1024, 3072)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 3072)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 3, 3072).view(1, 1, -1).expand(1, 1024, 3072)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 3072)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:self.dim]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[self.dim+2 + n_sigma:n_sigma + self.dim+3])  # 1 x h x w

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        return instance_map.unsqueeze(0)


class Cluster:

    def __init__(self, cls=False):

        xm = torch.linspace(0, 3, 3072).view(1, 1, -1).expand(1, 1024, 3072)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 3072)
        xym = torch.cat((xm, ym), 0)
        self.xym = xym.cuda()

        if cls:
            self.num_points = 1000
            # load model for FP and TP classification
            from models.BranchedERFNet import PointFeatMax
            model_path = os.path.join(rootDir, 'pointnet_finetune_FPNP/best_val_model.pth0.9169_0.7605') # avg TP: 0.8373, avg TP-FP: 0.6746
            model = PointFeatMax(ic=4, oc=2, num_points=1000)
            model = torch.nn.DataParallel(model).to(torch.device("cuda:0"))
            state = torch.load(model_path)
            new_state_dict = state['model_state_dict']
            model.load_state_dict(new_state_dict, strict=True)
            print('%s Loaded' % model_path)
            self.cls_model = model
            self.cls_model.eval()
            self.tp_thresh = 0.16
            # output = self.cls_model(pts.permute(0, 2, 1)) # FP0, TP1

    def cluster_with_gt(self, prediction, instance, n_sigma=1,):

        height, width = prediction.size(1), prediction.size(2)
    
        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
    
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
    
        instance_map = torch.zeros(height, width).byte().cuda()
    
        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]
    
        for id in unique_instances:
    
            mask = instance.eq(id).view(1, height, width)
    
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1
    
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s*10)  # n_sigma x 1 x 1
    
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2)*s, 0))
    
            proposal = (dist > 0.5)
            instance_map[proposal] = id
    
        return instance_map

    def compute_mask_iou(self, mask1, mask2):
        add = mask1.float() + mask2.float()
        return (add>1.0).sum().item() / float((add>0.0).sum().item())

    def cluster(self, prediction, n_sigma=1, threshold=0.5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2+n_sigma:2+n_sigma + 1])  # 1 x h x w
       
        instance_map = torch.zeros(height, width).byte()
        instances = []

        count = 1
        mask = seed_map > 0.5

        if mask.sum() > 128:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while(unclustered.sum() > 128):

                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed+1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed+1]*10)
                dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb_masked -
                                                        center, 2)*s, 0, keepdim=True))

                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > 128:
                    if unclustered[proposal].sum().float()/proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu()
                        instances.append({'mask': instance_mask.squeeze()*255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances

    def cluster_mots_wo_points(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, return_conf=False):
        confs = []
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        instance_map_masked[proposal.squeeze()] = count
                        confs.append(seed_score)
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        if return_conf:
            return instance_map, confs
        return instance_map

    def cluster_mots_w_cls(self, prediction, instances, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5):
        # collect dataset for FP and TP
        FPList, TPList = [], []
        gt_ids = torch.unique(instances)[1:]
        gt_masks = [instances.eq(el) for el in gt_ids]

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w
        cls_map = torch.sigmoid(prediction[3 + n_sigma:5 + n_sigma])  # 2 x h x w

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)
            cls_masked = cls_map[mask.expand_as(cls_map)].view(2, -1).permute(1,0)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                cls_score = cls_masked[seed:seed + 1]

                if seed_score < threshold or cls_score[0][1] < cls_score[0][0]:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        instance_map_masked[proposal.squeeze()] = count

                        # calculate cls variance
                        instance_map_bak = torch.zeros(height, width).byte()
                        instance_map_bak[mask.squeeze().cpu()] = instance_map_masked.cpu()
                        curr_mask = instance_map_bak == count

                        ious = [self.compute_mask_iou(curr_mask, el) for el in gt_masks]
                        # cls_list = cls_masked[proposal]
                        # vus = torch.nonzero(curr_mask).float()
                        # vus -= vus.mean(dim=0, keepdims=True)
                        # cls_list = torch.cat([vus.cpu(), cls_list.cpu()], dim=1).cpu().numpy()
                        if len(ious)==0 or max(ious) < 0.1:
                            # FP. record seed distribution
                            FPList.append(curr_mask.cpu().numpy())
                        elif max(ious) > 0.7 and seed_score < 0.98:
                            # TP. only consider bad TP
                            TPList.append(curr_mask.cpu().numpy())
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, FPList, TPList

    def cluster_mots_and_cls(self, prediction, instances, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5):
        # cluster with FP and TP
        gt_ids = torch.unique(instances)[1:]
        gt_masks = [instances.eq(el) for el in gt_ids]

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w
        cls_map = torch.sigmoid(prediction[3 + n_sigma:5 + n_sigma])  # 2 x h x w

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)
            cls_masked = cls_map[mask.expand_as(cls_map)].view(2, -1).permute(1,0)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                cls_score = cls_masked[seed:seed + 1]

                if seed_score < threshold or cls_score[0][1] < cls_score[0][0]:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        instance_map_masked[proposal.squeeze()] = count

                        # calculate cls variance
                        instance_map_bak = torch.zeros(height, width).byte()
                        instance_map_bak[mask.squeeze().cpu()] = instance_map_masked.cpu()
                        curr_mask = instance_map_bak == count

                        cls_list = cls_masked[proposal]
                        vus = torch.nonzero(curr_mask).float()
                        vus -= vus.mean(dim=0, keepdims=True)
                        pts = torch.cat([vus.cpu(), cls_list.cpu()], dim=1).cpu().numpy()
                        # pts processing
                        pts[:, 2] = np.clip(pts[:, 2], 1e-5, None)
                        choices = np.random.choice(pts.shape[0], self.num_points)
                        sel_pts = pts[choices].astype(np.float32)
                        sel_pts[:, 2] = sel_pts[:, 3] / sel_pts[:, 2]
                        sel_pts = torch.from_numpy(sel_pts[np.newaxis])

                        output = self.cls_model(sel_pts.permute(0, 2, 1))  # FP0, TP1
                        output = F.softmax(output, 1)
                        if output[0][1] < self.tp_thresh:
                            # False Positive: delete this inst
                            instance_map_masked[instance_map_masked==count] = 0
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map

    def cluster_mots_w_offset(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5):
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map

    def cluster_seed_cls(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None, return_conf=False):
        confs = []
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[4 + n_sigma:n_sigma + 5])  # 1 x h x w
        # seed_map = prediction[3 + n_sigma:n_sigma + 5]  # 1 x h x w
        # seed_map = F.softmax(seed_map, dim=0)[1:]

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            confs.append(seed_score)
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        if return_conf:
            return instance_map, confs
        return instance_map

    def cluster_seed_cls_shift(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None, return_conf=False):
        # shift to the center point
        confs = []
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        # seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w
        seed_map = torch.sigmoid(prediction[4 + n_sigma:n_sigma + 5])  # 1 x h x w
        # cls_map = torch.sigmoid(prediction[3 + n_sigma:n_sigma + 5]).permute(1, 2, 0)  # h x w x 2
        # seed_map = seed_map * cls_map[:, :, 1].unsqueeze(0)

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0   # set to zero to avoid dead circle
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))
                proposal = (dist > dist_thresh).squeeze().bool()

                # recompute center
                fg_vus = spatial_emb_masked.permute(1,0)[proposal]
                mean_vus = fg_vus.mean(0, keepdim=True)
                dist_proposal = ((fg_vus - mean_vus)**2).sum(1, keepdim=True)
                seed_in_proposal = dist_proposal.argmin().item()
                seed = torch.nonzero(proposal).squeeze(1)[seed_in_proposal].item()
                # seed_score = (seed_map_masked * unclustered.float()).squeeze(0)[seed]
                center = spatial_emb_masked[:, seed:seed + 1]
                # unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))
                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            # unclustered[proposal] = 0
                            pass
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            confs.append(seed_score)
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        if return_conf:
            return instance_map, confs
        return instance_map

    def cluster_seed_cls_with_scale(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, max_inst_pixel=1000000, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None, return_conf=False):
        confs = []
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        # seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w
        seed_map = torch.sigmoid(prediction[4 + n_sigma:n_sigma + 5])  # 1 x h x w
        # cls_map = torch.sigmoid(prediction[3 + n_sigma:n_sigma + 5]).permute(1, 2, 0)  # h x w x 2
        # seed_map = seed_map * cls_map[:, :, 1].unsqueeze(0)

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if min_inst_pixel < proposal.sum() < max_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            confs.append(seed_score)
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        if return_conf:
            return instance_map, confs
        return instance_map

    def cluster_seed_cls_w_dist(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None, return_conf=False):
        confs = []
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[4 + n_sigma:n_sigma + 5])  # 1 x h x w
        dist_map = torch.sigmoid(prediction[2 + n_sigma:3 + n_sigma])
        dist_map = (dist_map-0.05).clamp(max=0.7)
        dist_map[dist_map<dist_thresh] = dist_thresh

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)
            dist_map_masked = dist_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_map_masked).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            confs.append(seed_score)
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        if return_conf:
            return instance_map, confs
        return instance_map


class OffsetCluster:

    def __init__(self, cls=False):

        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        self.step = 1024.0
        xym = torch.cat((xm, ym), 0)
        self.xym = xym.cuda()

    def cluster(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        instance_map_masked[proposal.squeeze()] = count
                        # instance_mask = torch.zeros(height, width).byte()
                        # instance_mask[mask.squeeze().cpu()] = proposal.cpu()
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map

    def cluster_next_offset(self, prediction, offset_pred, initial_vus, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, initial_thresh=0.9):
        next_vus = []
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).type_as(prediction).byte()
        count = 1

        # initial instances by guidance. set initial UV seed to 1.0
        for vu in initial_vus:
            # check if out of scope
            if vu[1] < 0 or vu[1] >=width or vu[0] < 0 or vu[0] >=height:
                continue
            if seed_map[0,vu[0],vu[1]] > threshold*initial_thresh:
                seed_map[0, vu[0], vu[1]] = 1.0
        mask = seed_map > min_seed_thresh
        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > min_pixel):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked
        if count > 1:
            ids = torch.unique(instance_map)[1:]
            for id in ids:
                vus = torch.nonzero(instance_map == id).float()
                vu = torch.round(vus.mean(dim=0)).long()
                vu_offset = offset_pred[0, :, vu[0], vu[1]]
                next_vu = torch.round(vu+vu_offset).long()
                next_vus.append(next_vu)

        return instance_map.cpu(), next_vus

    def compute_offset_emb(self, instance_map, offset_pred):
        curr_pos, last_pos, masks = [], [], []
        ids = torch.unique(instance_map)
        if ids.shape[0] > 1:
            for id in ids[1:]:
                mask = instance_map == id
                masks.append(mask.cpu().numpy())
                vus = torch.nonzero(mask).float()
                curr_vu = vus.mean(dim=0).cpu().numpy()
                curr_pos.append(curr_vu)
                if offset_pred is not None:
                    cv, cu = int(curr_vu[0]), int(curr_vu[1])
                    ovu = offset_pred[0][:, cv, cu].cpu().numpy() * self.offsetMax
                    last_vu = curr_vu + ovu
                    last_pos.append(last_vu)
                else:
                    last_pos.append(curr_vu)
        return np.array(curr_pos), np.array(last_pos), masks


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)


class ClusterSeedClsWithFilter(nn.Module):
    # delete bad instances
    def __init__(self, dim=2, larger=False):
        super().__init__()

        if dim ==2:
            if larger:
                xm = torch.linspace(0, 3, 3072).view(1, 1, -1).expand(1, 2048, 3072)
                ym = torch.linspace(0, 2, 2048).view(1, -1, 1).expand(1, 2048, 3072)
            else:
                xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
                ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:self.dim]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        seed_map = prediction[self.dim+1 + n_sigma:n_sigma + self.dim+3]  # 1 x h x w
        seed_map = F.softmax(seed_map, dim=0)[1:]

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        instance_dict = {}
        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel) and count < 200:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))

                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            instance_dict[count] = proposal.sum().item()
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        # check inst pixel < min_inst_pixel
        ids = instance_map.unique()[1:]
        for id in ids:
            prev_pixels = instance_dict[id.item()]
            now_pixels = (instance_map==id).sum().item()
            if (not prev_pixels == now_pixels):
                if now_pixels < min_inst_pixel*3 or now_pixels/float(prev_pixels) < inst_ratio:
                    instance_map[instance_map==id] = 0
        return instance_map.unsqueeze(0)


class ClusterClsWithSeed(nn.Module):
    # delete bad instances
    def __init__(self, dim=2):
        super().__init__()

        if dim ==2:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:self.dim]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        seedVal = torch.sigmoid(prediction[self.dim + n_sigma:self.dim + n_sigma+1])  # 1 x h x w
        seed_map = prediction[self.dim+1 + n_sigma:n_sigma + self.dim+3]  # 1 x h x w
        seed_map = F.softmax(seed_map, dim=0)[1:]

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        instance_dict = {}
        if mask.sum() > min_pixel:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)
            seed_val_masked = seedVal[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel) and count < 200:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                unclustered[seed] = 0
                center = spatial_emb_masked[:, seed:seed + 1]
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))
                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    # recompute center and shift to the pixel with the highest seed value
                    seed_vals = seed_val_masked[0].clone()
                    seed_vals[~proposal] = 0.0
                    seed = seed_vals.argmax().item()
                    # seed_score = seed_map_masked[0,seed].item()
                    # if seed_score < threshold:
                    #     break
                    center = spatial_emb_masked[:, seed:seed + 1]
                    unclustered[seed] = 0
                    s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                    dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))
                    proposal = (dist > dist_thresh).squeeze().bool()

                    if proposal.sum() > min_inst_pixel:
                        if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                            seed_mean = seed_map_masked[0][proposal].mean()
                            if avg_seed is not None and seed_mean < avg_seed:
                                unclustered[proposal] = 0
                                continue
                            else:
                                instance_map_masked[proposal.squeeze()] = count
                                instance_dict[count] = proposal.sum().item()
                                count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        # check inst pixel < min_inst_pixel
        ids = instance_map.unique()[1:]
        for id in ids:
            prev_pixels = instance_dict[id.item()]
            now_pixels = (instance_map==id).sum().item()
            if (not prev_pixels == now_pixels):
                if now_pixels < min_inst_pixel*3 or now_pixels/float(prev_pixels) < inst_ratio:
                    instance_map[instance_map==id] = 0
        return instance_map.unsqueeze(0)


class ClusterSeedClsWithFilter0907(nn.Module):
    # delete bad instances
    def __init__(self, dim=2):
        super().__init__()

        if dim ==2:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            xym = torch.cat((xm, ym), 0)
        else: # dim=3
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
            dm = torch.ones(xm.shape).type_as(xm) * 0.5
            xym = torch.cat((xm, ym, dm), 0)
        self.xym = xym
        self.dim = dim

    def forward(self, prediction, n_sigma=2, threshold=0.5, min_pixel=160, min_inst_pixel=160, min_seed_thresh=0.5, inst_ratio=0.5, dist_thresh=0.5, avg_seed=None):
        prediction = prediction[0]
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:self.dim]) + xym_s.type_as(prediction)  # 2 x h x w
        sigma = prediction[self.dim:self.dim + n_sigma]  # n_sigma x h x w
        seed_map = prediction[self.dim+1 + n_sigma:n_sigma + self.dim+3]  # 1 x h x w
        seed_map = F.softmax(seed_map, dim=0)[1:]

        instance_map = torch.zeros(height, width).type_as(prediction).byte()

        count = 1
        mask = seed_map > min_seed_thresh

        instance_dict = {}
        if mask.sum() > min_pixel:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(self.dim, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).type_as(prediction).byte()
            instance_map_masked = torch.zeros(mask.sum()).type_as(prediction).byte()

            while (unclustered.sum() > min_pixel) and count < 200:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0, keepdim=True))
                dist = dist * seed_map_masked
                proposal = (dist > dist_thresh).squeeze().bool()

                if proposal.sum() > min_inst_pixel:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > inst_ratio:
                        seed_mean = seed_map_masked[0][proposal].mean()
                        if avg_seed is not None and seed_mean < avg_seed:
                            unclustered[proposal] = 0
                            continue
                        else:
                            instance_map_masked[proposal.squeeze()] = count
                            instance_dict[count] = proposal.sum().item()
                            count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        # check inst pixel < min_inst_pixel
        ids = instance_map.unique()[1:]
        for id in ids:
            prev_pixels = instance_dict[id.item()]
            now_pixels = (instance_map==id).sum().item()
            if (not prev_pixels == now_pixels):
                if now_pixels < min_inst_pixel*3 or now_pixels/float(prev_pixels) < inst_ratio:
                    instance_map[instance_map==id] = 0
        return instance_map.unsqueeze(0)


class ClusterOfficial:

    def __init__(self, ):

        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()

    def cluster_with_gt(self, prediction, instance, n_sigma=1, ):

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().cuda()

        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = instance.eq(id).view(1, height, width)

            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))

            proposal = (dist > 0.5)
            instance_map[proposal] = id

        return instance_map

    def cluster(self, prediction, n_sigma=1, threshold=0.5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).byte()
        instances = []

        count = 1
        mask = seed_map > 0.5

        if mask.sum() > 128:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            while (unclustered.sum() > 128):

                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, 0, keepdim=True))

                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > 128:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu()
                        instances.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances
