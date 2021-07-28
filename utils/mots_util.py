"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
from collections import namedtuple
import pycocotools.mask as maskUtils
import numpy as np
import cv2
import torch
import munkres
from scipy.spatial.distance import cdist
from config import *
from PIL import Image
import multiprocessing


def readInstCar(path, class_id=26):
    pic = np.array(Image.open(path))
    mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
    if mask.sum() > 0:
        return path
    else:
        return ''


def readInstPerson(path, class_id=2):
    pic = np.array(Image.open(path))
    mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
    if mask.sum() > 0:
        return path
    else:
        return ''


def leave_needed(inst_paths, class_id):
    # multiprocessing to filter out instances not including class_id

    pool = multiprocessing.Pool(processes=32)
    if class_id ==2:
        results = pool.map(readInstPerson, inst_paths)
    else:
        results = pool.map(readInstCar, inst_paths)
    pool.close()

    new_paths = []
    for path in results:
        if len(path)>0:
            new_paths.append(path)
    return new_paths


TrackElement = namedtuple("TrackElement", ["t", "track_id", "class_id", "mask", "embed", "points"])
munkres_obj = munkres.Munkres()
'''
p = subprocess.run(["python3", "eval.py",
                                "/home/voigtlaender/vision/savitar2/forwarded/" + self.config.string(
                                    "model") + "/tracking_data/",
                                "/globalwork/voigtlaender/data/KITTI_MOTS/train/instances/", "val.seqmap"],
                               stdout=subprocess.PIPE, cwd="/home/voigtlaender/vision/mots_eval/")
            print(p.stdout.decode("utf-8"), file=log.v1)
'''

def export_tracking_result_in_kitti_format(tag, tracks, add_masks, model_str, out_folder="", start_time_at_1=False):
    if out_folder == "":
        out_folder = "forwarded/" + model_str + "/tracking_data"
    os.makedirs(out_folder, exist_ok=True)
    out_filename = out_folder + "/" + tag + ".txt"
    with open(out_filename, "w") as f:
        start = 1 if start_time_at_1 else 0
        for t, tracks_t in enumerate(tracks, start):  # TODO this works?
            for track in tracks_t:
                if add_masks:
                    print(t, track.track_id, track.class_,
                          *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)
                else:
                    # TODO: class id to class name mapping atm harcoded
                    if track.class_ == 1:
                        class_str = "Car"
                    elif track.class_ == 2:
                        class_str = "Pedestrian"
                    else:
                        assert False, ("unknown class id", track.class_)
                    print(t, track.track_id, class_str, -1, -1, -1, *track.box, -1, -1, -1, -1, -1, -1, -1, track.score, file=f)


def newElem(t, track_id, embed, mask, points=None, class_id=1):
    mask_np = np.asfortranarray(mask.astype(np.uint8))
    mask_coco = maskUtils.encode(mask_np)
    return TrackElement(t=t, mask=mask_coco, class_id=class_id, track_id=track_id, embed=embed, points=points)


class TrackHelper(object):

    def __init__(self, save_dir, margin, t_car=0.8165986526897969, t_person=0.47985540892434836, alive_car=30, alive_person=30, car=True,
                 mask_iou=False, mask_iou_scale_person=0.54, mask_iou_scale_car=0.74, cosine=False, MOTS20=False):
        self.margin = margin # margin unused now
        self.save_dir = save_dir
        self.all_tracks = []
        self.active_tracks = []
        self.current_video= None
        self.next_inst_id = None
        self.mask_iou = mask_iou
        self.cosine = cosine
        if car:
            self.reid_euclidean_scale = 1.0090931467228708
            self.reid_euclidean_offset = 8.810218833503743
            self.association_threshold = t_car
            self.keep_alive = alive_car
            self.class_id=1
            self.mask_iou_scale = mask_iou_scale_car    # 0.74/1.0090931467228708
        else:
            self.class_id = 2
            if not MOTS20:
                # KITTI
                self.reid_euclidean_offset = 9.447084376750222
                self.reid_euclidean_scale = 1.3437965549876354
                self.association_threshold = t_person
                self.keep_alive = alive_person
                self.mask_iou_scale = mask_iou_scale_person # 0.54/1.3437965549876354
            else:
                # MOT
                self.reid_euclidean_offset = 9.447084376750222
                self.reid_euclidean_scale = 1.3437965549876354
                self.association_threshold = t_person
                self.keep_alive = 50
                self.mask_iou_scale = 0.01 # 0.54/1.3437965549876354

        print('params:', 'car' if car else 'pedestrian', self.keep_alive, self.association_threshold, 'mask_iou: %s' % self.mask_iou_scale if self.mask_iou else 'no mask iou')

    def reset(self, subfolder):
        self.all_tracks, self.active_tracks = [], []
        self.next_inst_id = self.class_id
        self.current_video = subfolder

    def export_last_video(self):
        print('Writing ', self.current_video)
        out_filename = os.path.join(self.save_dir, self.current_video + ".txt")
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        with open(out_filename, "w") as f:
            for track in self.all_tracks:
                print(track.t, track.track_id, track.class_id, *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)

    def compute_dist(self, embeds, embed):
        src = embed.unsqueeze(0)
        dist = torch.pow(src.repeat(embeds.shape[0], 1) - embeds, 2).sum(dim=1)
        return dist

    def update_active_track(self, frameCount):
        active_tracks_ = []
        for track in self.active_tracks:
            if track.t >= frameCount - self.keep_alive:
                active_tracks_.append(track)
        self.active_tracks = active_tracks_

    def tracking(self, subfolder, frameCount, embeds, masks):
        self.current_video = subfolder if self.current_video == None else self.current_video
        self.next_inst_id = self.class_id if self.next_inst_id == None else self.next_inst_id
        if not subfolder == self.current_video:
            # a new video
            self.export_last_video()
            self.reset(subfolder)

        self.update_active_track(frameCount)

        # traverse insts
        n = len(embeds)
        if n < 1:
            return
        if len(self.all_tracks) == 0 or len(self.active_tracks) == 0:
            # current frame is the first active one, so register all embeds
            for i in range(n):
                embed = embeds[i]
                mask = masks[i]
                current_inst = newElem(frameCount, self.next_inst_id, embed, mask, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 2
            return
        # compare inst by inst.
        # only compare with previous embeds, not including embeds of this frame
        last_reids = np.concatenate([el.embed[np.newaxis] for el in self.active_tracks], axis=0)
        not_last_frame = np.array([el.t < frameCount - 1 for el in self.active_tracks])
        curr_reids = embeds
        asso_sim = np.zeros((n, len(self.active_tracks)))

        detections_assigned = np.zeros(len(embeds)).astype(np.bool)
        reid_dists = cdist(curr_reids, last_reids, "euclidean" if not self.cosine else 'cosine')
        asso_sim += self.reid_euclidean_scale * (self.reid_euclidean_offset - reid_dists)

        if self.mask_iou:
            # consider add mask iou
            masks_t = [maskUtils.encode(np.asfortranarray(v.astype(np.uint8))) for v in masks]
            masks_tm1 = [v.mask for v in self.active_tracks]
            mask_ious = maskUtils.iou(masks_t, masks_tm1, [False] * len(masks_tm1))
            mask_ious[:, not_last_frame] = 0
            # try:
            asso_sim += self.mask_iou_scale * mask_ious
            # except:
            #     b=1

        cost_matrix = munkres.make_cost_matrix(asso_sim)
        # asso_sim[np.isnan(asso_sim)] = self.association_threshold*2
        disallow_indices = np.argwhere(asso_sim <= self.association_threshold)
        for ind in disallow_indices:
            cost_matrix[ind[0]][ind[1]] = 1e9
        indexes = munkres_obj.compute(cost_matrix)
        for row, column in indexes:
            value = cost_matrix[row][column]
            if value == 1e9:
                continue
            embed = embeds[row]
            mask = masks[row]
            current_inst = newElem(frameCount, self.active_tracks[column].track_id, embed, mask, class_id=self.class_id)
            self.all_tracks.append(current_inst)
            self.active_tracks[column] = current_inst
            detections_assigned[row] = True

        # new inst for unassigned
        detections_assigned_Inds = np.nonzero(detections_assigned == False)[0].tolist()
        if len(detections_assigned_Inds) > 0:
            for i in detections_assigned_Inds:
                embed = embeds[i]
                mask = masks[i]
                current_inst = newElem(frameCount, self.next_inst_id, embed, mask, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 2
        return


class TrackingOnlyHelper(object):

    def __init__(self, save_dir, margin, t_car=0.8165986526897969, t_person=0.47985540892434836, alive_car=5, alive_person=30, car=True,
                 mask_iou=False, mask_iou_scale_person=0.54, mask_iou_scale_car=0.74, cosine=False):
        # mask_iou_scale_car 0.7~0.8
        # t_car 0.7~1.0 or 6.0
        self.margin = margin
        self.save_dir = save_dir
        self.all_tracks = []
        self.active_tracks = []
        self.current_video= None
        self.next_inst_id = None
        self.mask_iou = mask_iou
        self.cosine = cosine
        if car:
            self.reid_euclidean_scale = 1.0090931467228708
            self.reid_euclidean_offset = 8.810218833503743
            self.association_threshold = t_car
            self.keep_alive = alive_car
            self.class_id=1
            self.mask_iou_scale = mask_iou_scale_car    # 0.74/1.0090931467228708
        else:
            self.reid_euclidean_offset = 9.447084376750222
            self.reid_euclidean_scale = 1.3437965549876354
            self.association_threshold = t_person
            self.keep_alive = alive_person
            self.class_id = 2
            self.mask_iou_scale = mask_iou_scale_person # 0.54/1.3437965549876354
        print('params:', 'car' if car else 'pedestrian', self.keep_alive, self.association_threshold, 'mask_iou: %s' % self.mask_iou_scale if self.mask_iou else 'no mask iou')

    def reset(self, subfolder):
        self.all_tracks, self.active_tracks = [], []
        self.next_inst_id = self.class_id
        self.current_video = subfolder

    def export_last_video(self, f):
        for track in self.all_tracks:
            print(track.t, track.track_id, track.class_id, *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)

    def compute_dist(self, embeds, embed):
        src = embed.unsqueeze(0)
        dist = torch.pow(src.repeat(embeds.shape[0], 1) - embeds, 2).sum(dim=1)
        return dist

    def update_active_track(self, frameCount):
        active_tracks_ = []
        for track in self.active_tracks:
            if track.t >= frameCount - self.keep_alive:
                active_tracks_.append(track)
        self.active_tracks = active_tracks_

    def tracking(self, subfolder, frameCount, embeds, masks):
        self.current_video = subfolder if self.current_video == None else self.current_video
        self.next_inst_id = self.class_id if self.next_inst_id == None else self.next_inst_id
        if not subfolder == self.current_video:
            # a new video
            self.reset(subfolder)

        self.update_active_track(frameCount)

        # traverse insts
        n = len(embeds)
        if n < 1:
            return
        if len(self.all_tracks) == 0 or len(self.active_tracks) == 0:
            # current frame is the first active one, so register all embeds
            for i in range(n):
                embed = embeds[i]
                mask = masks[i]
                current_inst = newElem(frameCount, self.next_inst_id, embed, mask, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 2
            return
        # compare inst by inst.
        # only compare with previous embeds, not including embeds of this frame
        last_reids = np.concatenate([el.embed[np.newaxis] for el in self.active_tracks], axis=0)
        not_last_frame = np.array([el.t < frameCount - 1 for el in self.active_tracks])
        curr_reids = embeds
        asso_sim = np.zeros((n, len(self.active_tracks)))

        detections_assigned = np.zeros(len(embeds)).astype(np.bool)
        reid_dists = cdist(curr_reids, last_reids, "euclidean" if not self.cosine else 'cosine')
        asso_sim += self.reid_euclidean_scale * (self.reid_euclidean_offset - reid_dists)

        if self.mask_iou:
            # consider add mask iou
            masks_t = [maskUtils.encode(np.asfortranarray(v.astype(np.uint8))) for v in masks]
            masks_tm1 = [v.mask for v in self.active_tracks]
            mask_ious = maskUtils.iou(masks_t, masks_tm1, [False] * len(masks_tm1))
            mask_ious[:, not_last_frame] = 0
            asso_sim += self.mask_iou_scale * mask_ious

        cost_matrix = munkres.make_cost_matrix(asso_sim)
        disallow_indices = np.argwhere(asso_sim <= self.association_threshold)
        for ind in disallow_indices:
            cost_matrix[ind[0]][ind[1]] = 1e9
        indexes = munkres_obj.compute(cost_matrix)

        for row, column in indexes:
            value = cost_matrix[row][column]
            if value == 1e9:
                continue
            embed = embeds[row]
            mask = masks[row]
            current_inst = newElem(frameCount, self.active_tracks[column].track_id, embed, mask, class_id=self.class_id)
            self.all_tracks.append(current_inst)
            self.active_tracks[column] = current_inst
            detections_assigned[row] = True

        # new inst for unassigned
        detections_assigned_Inds = np.nonzero(detections_assigned == False)[0].tolist()
        if len(detections_assigned_Inds) > 0:
            for i in detections_assigned_Inds:
                embed = embeds[i]
                mask = masks[i]
                current_inst = newElem(frameCount, self.next_inst_id, embed, mask, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 2
        return


def getMasksFromInstanceMap(instMap):
    uids = np.unique(instMap)[1:]
    masks = []
    for uid in uids:
        masks.append(np.array(instMap==uid))
    return masks


class TrackHelper0226(object):
    # add border rejection
    def __init__(self, save_dir, margin, t_car=0.8165986526897969, t_person=0.47985540892434836, alive_car=8, alive_person=10, car=True,
                 mask_iou=True, mask_iou_scale_person=0.54, mask_iou_scale_car=0.74, cosine=False, MOTS20=False, test=False):
        self.margin = margin # margin unused now
        self.save_dir = save_dir
        self.all_tracks = []
        self.active_tracks = []
        self.current_video= None
        self.next_inst_id = None
        self.mask_iou = mask_iou
        self.cosine = cosine
        if car:
            self.keep_border_alive = 3
            self.border_threhold = 1.5
            self.border_ratio = 0.1
            self.reid_euclidean_scale = 1.0090931467228708
            self.reid_euclidean_offset = 8.810218833503743
            self.association_threshold = t_car
            self.keep_alive = alive_car
            self.class_id=1
            self.mask_iou_scale = mask_iou_scale_car    # 0.74/1.0090931467228708
            self.image_root = os.path.join(kittiRoot, 'images') if not test else os.path.join(kittiRoot, 'testing/image_02/')
            self.suffix = '.png'
        else:
            self.class_id = 2
            self.keep_border_alive = 2  # 2 is better for sMOTSA, 1 is better for HOTA
            self.border_threhold = 1.0
            self.border_ratio = 0.05
            if not MOTS20:
                # KITTI
                self.reid_euclidean_offset = 9.447084376750222
                self.reid_euclidean_scale = 1.3437965549876354
                self.association_threshold = t_person
                self.keep_alive = alive_person
                self.mask_iou_scale = mask_iou_scale_person # 0.54/1.3437965549876354
                self.image_root = os.path.join(kittiRoot, 'images') if not test else os.path.join(kittiRoot, 'testing/image_02/')
                self.suffix = '.png'
            else:
                # MOT
                self.reid_euclidean_offset = 9.447084376750222
                self.reid_euclidean_scale = 1.3437965549876354
                self.association_threshold = t_person
                self.keep_alive = alive_person
                self.mask_iou_scale = mask_iou_scale_person # 0.54/1.3437965549876354
                self.image_root = os.path.join(MOTS20Root, 'images')
                self.suffix = '.jpg'

        print('params:', 'car' if car else 'pedestrian', self.keep_alive, self.association_threshold, 'mask_iou: %s' % self.mask_iou_scale if self.mask_iou else 'no mask iou')

    def reset(self, subfolder):
        self.all_tracks, self.active_tracks = [], []
        self.next_inst_id = 1
        self.current_video = subfolder

    def export_last_video(self):
        print('Writing ', self.current_video)
        out_filename = os.path.join(self.save_dir, self.current_video + ".txt")
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        with open(out_filename, "w") as f:
            for track in self.all_tracks:
                print(track.t, track.track_id, track.class_id, *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)

    def compute_dist(self, embeds, embed):
        src = embed.unsqueeze(0)
        dist = torch.pow(src.repeat(embeds.shape[0], 1) - embeds, 2).sum(dim=1)
        return dist

    def update_active_track(self, frameCount):
        active_tracks_ = []
        for track in self.active_tracks:
            keep_alive = self.keep_alive if not track.points else self.keep_border_alive
            if track.t >= frameCount - keep_alive:
                active_tracks_.append(track)
        self.active_tracks = active_tracks_

    def newElemPos(self, t, track_id, embed, mask, img_size, points=None, class_id=1):  # with position record, points represent near border or not
        mask_np = np.asfortranarray(mask.astype(np.uint8))
        mask_coco = maskUtils.encode(mask_np)
        x, y, w, h = maskUtils.toBbox(mask_coco)
        cx, cy = x + w / 2, y + h / 2
        x1, x2 = x, x + w
        W, H = img_size
        border_dis = min(cx / w, (W - cx) / w) < self.border_threhold
        near_border = (x1 < W * self.border_ratio) or (x2 > W * (1-self.border_ratio))
        return TrackElement(t=t, mask=mask_coco, class_id=class_id, track_id=track_id, embed=embed, points=border_dis or near_border)

    def tracking(self, subfolder, frameCount, embeds, masks):
        self.current_video = subfolder if self.current_video == None else self.current_video
        self.next_inst_id = 1 if self.next_inst_id == None else self.next_inst_id
        if not subfolder == self.current_video:
            # a new video
            self.export_last_video()
            self.reset(subfolder)

        if len(masks) > 0:
            h, w = masks[0].shape
            img_size = (w, h)
        else:
            img_size = Image.open(os.path.join(self.image_root, subfolder, '%06d'%frameCount + self.suffix)).size
        self.update_active_track(frameCount)

        # traverse insts
        n = len(embeds)
        if n < 1:
            return
        if len(self.all_tracks) == 0 or len(self.active_tracks) == 0:
            # current frame is the first active one, so register all embeds
            for i in range(n):
                embed = embeds[i]
                mask = masks[i]
                current_inst = self.newElemPos(frameCount, self.next_inst_id, embed, mask, img_size, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 1
            return
        # compare inst by inst.
        # only compare with previous embeds, not including embeds of this frame
        last_reids = np.concatenate([el.embed[np.newaxis] for el in self.active_tracks], axis=0)
        not_last_frame = np.array([el.t < frameCount - 1 for el in self.active_tracks])
        curr_reids = embeds
        asso_sim = np.zeros((n, len(self.active_tracks)))

        detections_assigned = np.zeros(len(embeds)).astype(np.bool)
        reid_dists = cdist(curr_reids, last_reids, "euclidean" if not self.cosine else 'cosine')
        asso_sim += self.reid_euclidean_scale * (self.reid_euclidean_offset - reid_dists)

        if self.mask_iou:
            # consider add mask iou
            masks_t = [maskUtils.encode(np.asfortranarray(v.astype(np.uint8))) for v in masks]
            masks_tm1 = [v.mask for v in self.active_tracks]
            mask_ious = maskUtils.iou(masks_t, masks_tm1, [False] * len(masks_tm1))
            mask_ious[:, not_last_frame] = 0
            # try:
            asso_sim += self.mask_iou_scale * mask_ious
            # except:
            #     b=1

        cost_matrix = munkres.make_cost_matrix(asso_sim)
        # asso_sim[np.isnan(asso_sim)] = self.association_threshold*2
        disallow_indices = np.argwhere(asso_sim <= self.association_threshold)
        for ind in disallow_indices:
            cost_matrix[ind[0]][ind[1]] = 1e9
        indexes = munkres_obj.compute(cost_matrix)
        for row, column in indexes:
            value = cost_matrix[row][column]
            if value == 1e9:
                continue
            embed = embeds[row]
            mask = masks[row]
            current_inst = self.newElemPos(frameCount, self.active_tracks[column].track_id, embed, mask, img_size, class_id=self.class_id)
            self.all_tracks.append(current_inst)
            self.active_tracks[column] = current_inst
            detections_assigned[row] = True

        # new inst for unassigned
        detections_assigned_Inds = np.nonzero(detections_assigned == False)[0].tolist()
        if len(detections_assigned_Inds) > 0:
            for i in detections_assigned_Inds:
                embed = embeds[i]
                mask = masks[i]
                current_inst = self.newElemPos(frameCount, self.next_inst_id, embed, mask, img_size, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 1
        return


class TrackingOnlyHelper0226(object):

    def __init__(self, save_dir, margin, t_car=0.8165986526897969, t_person=0.47985540892434836, alive_car=8, alive_person=10, car=True,
                 mask_iou=True, mask_iou_scale_person=0.54, mask_iou_scale_car=0.74, cosine=False, MOTS20=False):
        self.margin = margin # margin unused now
        self.save_dir = save_dir
        self.all_tracks = []
        self.active_tracks = []
        self.current_video= None
        self.next_inst_id = None
        self.mask_iou = mask_iou
        self.cosine = cosine
        self.image_root = os.path.join(kittiRoot, 'testing/image_02/')
        self.suffix = '.png'
        if car:
            self.keep_border_alive = 2
            self.border_threhold = 1.5
            self.border_ratio = 0.1
            self.reid_euclidean_scale = 1.0090931467228708
            self.reid_euclidean_offset = 8.810218833503743
            self.association_threshold = t_car
            self.keep_alive = alive_car
            self.class_id=1
            self.mask_iou_scale = mask_iou_scale_car    # 0.74/1.0090931467228708
        else:
            self.keep_border_alive = 1
            self.border_threhold = 2.0
            self.border_ratio = 0.1
            self.reid_euclidean_offset = 9.447084376750222
            self.reid_euclidean_scale = 1.3437965549876354
            self.association_threshold = t_person
            self.keep_alive = alive_person
            self.class_id = 2
            self.mask_iou_scale = mask_iou_scale_person # 0.54/1.3437965549876354
        print('params:', 'car' if car else 'pedestrian', self.keep_alive, self.association_threshold, 'mask_iou: %s' % self.mask_iou_scale if self.mask_iou else 'no mask iou')

    def reset(self, subfolder):
        self.all_tracks, self.active_tracks = [], []
        self.next_inst_id = self.class_id
        self.current_video = subfolder

    def export_last_video(self, f):
        for track in self.all_tracks:
            print(track.t, track.track_id, track.class_id, *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)

    def compute_dist(self, embeds, embed):
        src = embed.unsqueeze(0)
        dist = torch.pow(src.repeat(embeds.shape[0], 1) - embeds, 2).sum(dim=1)
        return dist

    def update_active_track(self, frameCount):
        active_tracks_ = []
        for track in self.active_tracks:
            keep_alive = self.keep_alive if not track.points else self.keep_border_alive
            if track.t >= frameCount - keep_alive:
                active_tracks_.append(track)
        self.active_tracks = active_tracks_

    def newElemPos(self, t, track_id, embed, mask, img_size, points=None, class_id=1):  # with position record, points represent near border or not
        mask_np = np.asfortranarray(mask.astype(np.uint8))
        mask_coco = maskUtils.encode(mask_np)
        x, y, w, h = maskUtils.toBbox(mask_coco)
        cx, cy = x + w / 2, y + h / 2
        x1, x2 = x, x + w
        W, H = img_size
        border_dis = min(cx / w, (W - cx) / w) < self.border_threhold
        near_border = (x1 < W * self.border_ratio) or (x2 > W * (1-self.border_ratio))
        return TrackElement(t=t, mask=mask_coco, class_id=class_id, track_id=track_id, embed=embed, points=border_dis or near_border)

    def tracking(self, subfolder, frameCount, embeds, masks):
        self.current_video = subfolder if self.current_video == None else self.current_video
        self.next_inst_id = self.class_id if self.next_inst_id == None else self.next_inst_id
        if not subfolder == self.current_video:
            # a new video
            self.reset(subfolder)

        if len(masks) > 0:
            h, w = masks[0].shape
            img_size = (w, h)
        else:
            img_size = Image.open(os.path.join(self.image_root, subfolder, '%06d'%frameCount + self.suffix)).size
        self.update_active_track(frameCount)

        # traverse insts
        n = len(embeds)
        if n < 1:
            return
        if len(self.all_tracks) == 0 or len(self.active_tracks) == 0:
            # current frame is the first active one, so register all embeds
            for i in range(n):
                embed = embeds[i]
                mask = masks[i]
                current_inst = self.newElemPos(frameCount, self.next_inst_id, embed, mask, img_size, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 2
            return
        # compare inst by inst.
        # only compare with previous embeds, not including embeds of this frame
        last_reids = np.concatenate([el.embed[np.newaxis] for el in self.active_tracks], axis=0)
        not_last_frame = np.array([el.t < frameCount - 1 for el in self.active_tracks])
        curr_reids = embeds
        asso_sim = np.zeros((n, len(self.active_tracks)))

        detections_assigned = np.zeros(len(embeds)).astype(np.bool)
        reid_dists = cdist(curr_reids, last_reids, "euclidean" if not self.cosine else 'cosine')
        asso_sim += self.reid_euclidean_scale * (self.reid_euclidean_offset - reid_dists)

        if self.mask_iou:
            # consider add mask iou
            masks_t = [maskUtils.encode(np.asfortranarray(v.astype(np.uint8))) for v in masks]
            masks_tm1 = [v.mask for v in self.active_tracks]
            mask_ious = maskUtils.iou(masks_t, masks_tm1, [False] * len(masks_tm1))
            mask_ious[:, not_last_frame] = 0
            asso_sim += self.mask_iou_scale * mask_ious

        cost_matrix = munkres.make_cost_matrix(asso_sim)
        disallow_indices = np.argwhere(asso_sim <= self.association_threshold)
        for ind in disallow_indices:
            cost_matrix[ind[0]][ind[1]] = 1e9
        indexes = munkres_obj.compute(cost_matrix)

        for row, column in indexes:
            value = cost_matrix[row][column]
            if value == 1e9:
                continue
            embed = embeds[row]
            mask = masks[row]
            current_inst = self.newElemPos(frameCount, self.active_tracks[column].track_id, embed, mask, img_size, class_id=self.class_id)
            self.all_tracks.append(current_inst)
            self.active_tracks[column] = current_inst
            detections_assigned[row] = True

        # new inst for unassigned
        detections_assigned_Inds = np.nonzero(detections_assigned == False)[0].tolist()
        if len(detections_assigned_Inds) > 0:
            for i in detections_assigned_Inds:
                embed = embeds[i]
                mask = masks[i]
                current_inst = self.newElemPos(frameCount, self.next_inst_id, embed, mask, img_size, class_id=self.class_id)
                self.all_tracks.append(current_inst)
                self.active_tracks.append(current_inst)
                self.next_inst_id += 2
        return


def plot_instanceMap(inst_np, bgWhite=False):
    dColors = [(128, 0, 0), (170, 110, 40), (128, 128, 0), (0, 128, 128), (0, 0, 128), (230, 25, 75), (245, 130, 48)
        , (255, 225, 25), (210, 245, 60), (60, 180, 75), (70, 240, 240), (0, 130, 200), (145, 30, 180), (240, 50, 230)]
    unique_instances = np.unique(inst_np[inst_np>0])
    masks = [inst_np == id for id in unique_instances]
    if not bgWhite:
        FN = np.zeros((inst_np.shape[0], inst_np.shape[1], 3))
    else:
        FN = np.ones((inst_np.shape[0], inst_np.shape[1], 3)) * 255
    for im, mask_ in enumerate(masks):
        FN[mask_] = np.array(dColors[(im % len(dColors[1:])) + 1])
    return FN