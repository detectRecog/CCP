"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math

import numpy as np

import torch
import torch.nn as nn
from criterions.lovasz_losses import lovasz_hinge
import torch.nn.functional as F


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


class MOTSSegClsLoss0904(nn.Module):

    def __init__(self, to_center=True, n_sigma=2, foreground_weight=200, cls_ratio=1.0, focus_ratio=2.0, eval=False):
        super().__init__()
        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}, cls_ratio: {}, focus_ratio: {}'.format(
            to_center, n_sigma, foreground_weight, cls_ratio, focus_ratio))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # coordinate map
        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)
        self.cls_criterion = nn.SmoothL1Loss(reduction='sum')
        # self.cls_criterion = nn.CrossEntropyLoss(ignore_index=254, reduction='mean')

        from criterions.ghm_loss import FocalLoss, FocalOHEMLoss
        # self.ghm_cls = FocalLoss(balance_param=1.0, focusing_param=focus_ratio)
        self.ghm_cls = FocalOHEMLoss(balance_param=1.0, focusing_param=focus_ratio)
        # self.ghm_cls = GHMC()
        self.cls_ratio = cls_ratio
        if eval:
            self.ghm_cls = FocalLoss(balance_param=1.0, focusing_param=focus_ratio)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, seed_w=None, iou=False, iou_meter=None, show_seed=False):

        batch_size, height, width = prediction.size(0), prediction.size(2), prediction.size(3)
        if seed_w is None:
            seed_w = torch.ones(batch_size).type_as(prediction)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0
        cls_loss = 0

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2+self.n_sigma]  # n_sigma x h x w
            seed_map = torch.sigmoid(prediction[b, 2+self.n_sigma:2+self.n_sigma + 1])  # 1 x h x w
            cls_map = prediction[b, 3+self.n_sigma:self.n_sigma + 5].permute(1,2,0)  # h x w x 2
            # cls_map = torch.sigmoid(prediction[b, 3+self.n_sigma:self.n_sigma + 5]).permute(1,2,0)  # h x w x 2

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x h x w
            label = labels[b].unsqueeze(0)  # 1 x h x w
            valid_label = label[0] < 2

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # select valid pixels (exclude dontcare)
            cls_map_valid, label_valid = cls_map[valid_label], label[0][valid_label]
            cls_loss += self.cls_ratio * self.ghm_cls(cls_map_valid, label_valid.long()) * seed_w[b]

            # bg seed loss: regress bg to zero
            bg_mask = label == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            # if not iou:
            #     # not car to zero
            #     bg_mask = label == -1
            #     if bg_mask.sum() > 0:
            #         seed_loss += torch.sum(torch.pow(seed_map[bg_mask] - 0, 2)) * 10

            for id in instance_ids:

                in_mask = instance.eq(id)   # 1 x h x w

                # calculate center of attraction
                if self.to_center:
                    xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # calculate sigma (radius)
                sigma_in = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1)

                s = sigma_in.mean(1).view(self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                # sigma (radius) variation loss: calculate var loss before exp
                var_loss = var_loss + torch.mean(torch.pow(sigma_in.unsqueeze(1) - s.detach(), 2))

                s = torch.exp(s*10)

                # calculate gaussian
                dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb - center, 2)*s, 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + lovasz_hinge(dist*2-1, in_mask)

                # seed loss
                seed_loss += self.foreground_weight * torch.sum(torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += (w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss) * seed_w[b]
            # loss *= seed_w[b] # bug

        loss = loss / (b+1) + cls_loss / (b+1)
        focal_loss = cls_loss / (b+1)

        if show_seed:
            return loss + prediction.sum()*0, focal_loss
        else:
            return loss + prediction.sum() * 0


class MOTSSegClsRecLoss0131Lightn(nn.Module):
    # with reconstruction
    def __init__(self, to_center=True, n_sigma=2, foreground_weight=200, cls_ratio=1.0, focus_ratio=2.0, fg_center=False, mot17=False):
        super().__init__()
        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}, cls_ratio: {}, focus_ratio: {}, fg_center: {}'.format(
            to_center, n_sigma, foreground_weight, cls_ratio, focus_ratio, fg_center))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        self.fg_center = fg_center

        # coordinate map
        if mot17:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 2048, 2048)
            ym = torch.linspace(0, 2, 2048).view(1, -1, 1).expand(1, 2048, 2048)
        else:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)
        self.cls_criterion = nn.SmoothL1Loss(reduction='sum')
        # self.cls_criterion = nn.CrossEntropyLoss(ignore_index=254, reduction='mean')

        from criterions.ghm_loss import FocalLoss, FocalOHEMLoss
        self.ghm_cls = FocalOHEMLoss(balance_param=1.0, focusing_param=focus_ratio)
        self.cls_ratio = cls_ratio

    def forward(self, prediction, ims, instances, labels, w_inst=1, w_var=10, w_seed=1, seed_w=None, iou=False, max_instance_num=25):

        batch_size, height, width = prediction.size(0), prediction.size(2), prediction.size(3)
        if seed_w is None:
            seed_w = torch.ones(batch_size).type_as(prediction)
        # elif isinstance(seed_w, float):
        #     seed_w = torch.ones(batch_size).type_as(prediction) * seed_w
        # else:
        #     pass

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0
        res_loss = 0
        cls_loss = 0
        cls_count = 0
        iou_meter = []
        if instances.sum() > 0:
            rec_ims = prediction[:,-3:]
            pixel_diff= (ims.permute(0,2,3,1)[instances>0] - rec_ims.permute(0,2,3,1)[instances>0]) ** 2
            res_loss += pixel_diff.mean()

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2+self.n_sigma]  # n_sigma x h x w
            seed_map = torch.sigmoid(prediction[b, 2+self.n_sigma:2+self.n_sigma + 1])  # 1 x h x w
            cls_map = prediction[b, 3+self.n_sigma:self.n_sigma + 5].permute(1,2,0)  # h x w x 2
            # cls_map = torch.sigmoid(prediction[b, 3+self.n_sigma:self.n_sigma + 5]).permute(1,2,0)  # h x w x 2

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x h x w
            label = labels[b].unsqueeze(0)  # 1 x h x w
            valid_label = label[0] < 2

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # select valid pixels (exclude dontcare)
            if valid_label.sum() > 0:
                cls_map_valid, label_valid = cls_map[valid_label], label[0][valid_label]
                cls_loss += self.cls_ratio * self.ghm_cls(cls_map_valid, label_valid.long()) * seed_w[b]
                cls_count += 1

            # bg seed loss: regress bg to zero
            bg_mask = label == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            if instance_ids.shape[0] > 0:
                instance_ids = instance_ids[torch.randperm(instance_ids.shape[0])][:max_instance_num]
                for id in instance_ids:

                    in_mask = instance.eq(id)   # 1 x h x w

                    # calculate center of attraction
                    if self.fg_center:
                        with torch.no_grad():
                            xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                            ptsSel = xy_in[:, torch.randperm(xy_in.shape[1])[:max(min(xy_in.shape[1]//2, 3000), 1)]]
                            diff = ptsSel.unsqueeze(1) - ptsSel.unsqueeze(2)
                            dist = (diff[0]**2+diff[1]**2).sum(-1)
                            center = xy_in[:, dist.argmin()].unsqueeze(-1).unsqueeze(-1)
                    elif self.to_center:
                        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                        center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                    else:
                        center = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                    # calculate sigma (radius)
                    sigma_in = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1)

                    s_ = sigma_in.mean(1).view(self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                    s = torch.exp(s_*10)

                    # calculate gaussian
                    dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb - center, 2)*s, 0, keepdim=True))

                    # apply lovasz-hinge loss
                    instance_loss += lovasz_hinge(dist * 2 - 1, in_mask)
                    # sigma (radius) variation loss: calculate var loss before exp
                    var_loss = var_loss + torch.mean(torch.pow(sigma_in.unsqueeze(1) - s_.detach(), 2))

                    # seed loss
                    seed_loss += self.foreground_weight * torch.sum(
                        torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                    # calculate instance iou
                    if iou:
                        iou_meter.append(calculate_iou(dist.detach() > 0.5, in_mask))
                    obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += (w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss) * seed_w[b]
            if torch.isnan(instance_loss+var_loss+seed_loss+cls_loss):
                b=1
            # loss *= seed_w[b] # bug

        loss = loss / (b+1) + cls_loss / (b+1) + res_loss * 0.1
        focal_loss = cls_loss / (b+1)

        return loss + prediction.sum() * 0, focal_loss, iou_meter


class MOTSSegClsRecLoss0226Lightn(nn.Module):
    # support self-training
    def __init__(self, to_center=True, n_sigma=2, foreground_weight=200, cls_ratio=1.0, focus_ratio=2.0, fg_center=False, mot17=False, max_inst=25):
        super().__init__()
        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}, cls_ratio: {}, focus_ratio: {}, fg_center: {}'.format(
            to_center, n_sigma, foreground_weight, cls_ratio, focus_ratio, fg_center))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        self.fg_center = fg_center

        # coordinate map
        if mot17:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 2048, 2048)
            ym = torch.linspace(0, 2, 2048).view(1, -1, 1).expand(1, 2048, 2048)
        else:
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)
        self.cls_criterion = nn.SmoothL1Loss(reduction='sum')
        # self.cls_criterion = nn.CrossEntropyLoss(ignore_index=254, reduction='mean')

        from criterions.ghm_loss import FocalLoss, FocalOHEMLoss
        self.ghm_cls = FocalOHEMLoss(balance_param=1.0, focusing_param=focus_ratio)
        self.cls_ratio = cls_ratio
        self.max_inst = max_inst

    def forward(self, prediction, ims, instances, labels, w_inst=1, w_var=10, w_seed=1, seed_w=None, iou=False, is_test=False):

        batch_size, height, width = prediction.size(0), prediction.size(2), prediction.size(3)
        if seed_w is None:
            seed_w = torch.ones(batch_size).type_as(prediction)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0
        res_loss = 0
        cls_loss = 0
        cls_count = 0
        iou_meter = []
        if instances.sum() > 0:
            rec_ims = prediction[:,-3:]
            pixel_diff= (ims.permute(0,2,3,1)[instances>0] - rec_ims.permute(0,2,3,1)[instances>0]) ** 2
            res_loss += pixel_diff.mean()

        for b in range(0, batch_size):
            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2+self.n_sigma]  # n_sigma x h x w
            seed_map = torch.sigmoid(prediction[b, 2+self.n_sigma:2+self.n_sigma + 1])  # 1 x h x w
            cls_map = prediction[b, 3+self.n_sigma:self.n_sigma + 5].permute(1,2,0)  # h x w x 2

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x h x w
            label = labels[b].unsqueeze(0)  # 1 x h x w
            if is_test:
                # regard 预测值<0.01的位置认为是bg
                pred_cls_map = torch.nn.functional.softmax(prediction[b, 3+self.n_sigma:self.n_sigma + 5], dim=0)[1:]
                label[pred_cls_map<0.01] = 0
            valid_label = label[0] < 2

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # select valid pixels (exclude dontcare)
            if valid_label.sum() > 0:
                cls_map_valid, label_valid = cls_map[valid_label], label[0][valid_label]
                cls_loss += self.cls_ratio * self.ghm_cls(cls_map_valid, label_valid.long()) * seed_w[b]
                cls_count += 1

            # bg seed loss: regress bg to zero
            bg_mask = label == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            if instance_ids.shape[0] > 0:
                instance_ids = instance_ids[torch.randperm(instance_ids.shape[0])][:self.max_inst]
                for id in instance_ids:

                    in_mask = instance.eq(id)   # 1 x h x w

                    # calculate center of attraction
                    if self.fg_center:
                        with torch.no_grad():
                            xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                            ptsSel = xy_in[:, torch.randperm(xy_in.shape[1])[:max(min(xy_in.shape[1]//2, 3000), 1)]]
                            diff = ptsSel.unsqueeze(1) - ptsSel.unsqueeze(2)
                            dist = (diff[0]**2+diff[1]**2).sum(-1)
                            center = xy_in[:, dist.argmin()].unsqueeze(-1).unsqueeze(-1)
                    elif self.to_center:
                        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                        center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                    else:
                        center = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                    # calculate sigma (radius)
                    sigma_in = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1)

                    s_ = sigma_in.mean(1).view(self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                    s = torch.exp(s_*10)

                    # calculate gaussian
                    dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb - center, 2)*s, 0, keepdim=True))

                    # apply lovasz-hinge loss
                    instance_loss += lovasz_hinge(dist * 2 - 1, in_mask)
                    # sigma (radius) variation loss: calculate var loss before exp
                    var_loss = var_loss + torch.mean(torch.pow(sigma_in.unsqueeze(1) - s_.detach(), 2))

                    # seed loss
                    seed_loss += self.foreground_weight * torch.sum(
                        torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                    # calculate instance iou
                    if iou:
                        iou_meter.append(calculate_iou(dist.detach() > 0.5, in_mask))
                    obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += (w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss) * seed_w[b]
            if torch.isnan(instance_loss+var_loss+seed_loss+cls_loss):
                print('nan')
            # loss *= seed_w[b] # bug

        loss = loss / (b+1) + cls_loss / (b+1) + res_loss * 0.1
        focal_loss = cls_loss / (b+1)

        return loss + prediction.sum() * 0, focal_loss, iou_meter

