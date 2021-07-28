import glob
import os
import random
import numpy as np
import math
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import cv2
from config import *
from file_utils import *
from utils.mots_util import *
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageStat


class MOTSContinousCopyAndPasteCarsV3(Dataset):
    # Demo code of CCP for a single class: car (class_id=1), person (class_id=2).
    # KITTI MOTS train/val/test splits
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}
    SEQ_IDS_TEST = ["%04d" % idx for idx in range(29)]
    TIMESTEPS_PER_SEQ_TEST = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
                              '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
                              '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
                              '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
                              '0002': 243}
    def __init__(self, root_dir='./', type="train", size=None, transform=None, class_id=1, img_size=(320, 1088),
                 add_kins=True, centerRadius=False, scale_range = (0.5, 2.0), ori_prob = 0.2, requireObject=False, check_paste_quality=False, non_shift=True, min_inst_num=2,
                 max_instance_num = 15, nearby = 7, jitter=False, border_aug=False, randomCropShift=False, paste_by_order=False):

        self.class_id = class_id
        self.img_size = img_size    # (h,w)
        self.img_sizeT = (img_size[1], img_size[0])    # (w, h)
        self.type = type
        self.pad_size = (384, 1248)
        self.centerRadius = centerRadius # compute center and radius for each instance
        self.requireObject = requireObject # require at least one FG object
        self.check_paste_quality = check_paste_quality # check paste object quality
        self.min_inst_pixel = 64 # min paste pixel
        self.min_inst_num = min_inst_num # min instance number for training metric learning
        self.non_shift = non_shift # not shift position
        self.nearby = nearby
        self.jitter = jitter
        self.paste_by_order = paste_by_order
        self.border_aug = border_aug # aug inst on the left/right border, should be used together with non_shift
        self.randomCropShift = randomCropShift # random crop inst

        self.mots_instance_root = os.path.join(kittiRoot, 'instances')
        self.mots_image_root = os.path.join(kittiRoot, 'images')
        if type == 'trainval':
            mots_persons = load_pickle(os.path.join(kittiRoot, 'mots_inst_train5.pkl')) # train + 50% val
            print("train for testset models!")
        elif type == 'test':
            self.squence = self.SEQ_IDS_TEST
            self.mots_image_root = os.path.join(kittiRoot, 'testing/image_02')
            mots_persons = []
            for subdir in self.squence:
                image_list = sorted(make_dataset(os.path.join(self.mots_image_root, subdir), suffix='.png'))
                mots_persons += image_list
            self.mots_instance_list = mots_persons
            self.mots_image_list = mots_persons
        else:
            self.squence = self.SEQ_IDS_TRAIN if type in 'training' else self.SEQ_IDS_VAL
            mots_persons = []
            for subdir in self.squence:
                instance_list = sorted(make_dataset(os.path.join(self.mots_instance_root, subdir), suffix='.png'))
                instance_list = ['/'.join(el.split('/')[-2:]) for el in instance_list]
                for i in instance_list:
                    mots_persons.append(i)
        self.mots_instance_list = [os.path.join(self.mots_instance_root, el) for el in mots_persons]
        self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]

        if add_kins and (not type in ['test', 'val']):
            dst_kins = 'person_KINS' if self.class_id == 2 else 'KINS'
            instance_list = make_dataset(os.path.join(kittiRoot, 'training/' + dst_kins), suffix='.png') + make_dataset(
                os.path.join(kittiRoot, 'testing/' + dst_kins), suffix='.png')
            image_list = [el.replace(dst_kins, 'image_2') for el in instance_list]
            print('add KINS: ', len(image_list))
            self.mots_image_list += image_list
            self.mots_instance_list += instance_list
        else:
            print('wo KINS! for finetune')

        # load instanceSegDB which stores all instance blocks from the training set.
        if self.class_id == 1:
            self.instancesSegDB = load_pickle(os.path.join(kittiRoot, 'instanceSegDB_car_train.pkl'))
        else:
            self.instancesSegDB = load_pickle(os.path.join(kittiRoot, 'instanceSegDB_person_train.pkl'))
        self.instancesKeys = list(self.instancesSegDB.keys())
        self.real_size = len(self.mots_image_list)
        self.size = size
        self.transform = transform

        print('MOTSContinousCopyAndPasteCarsV3 Dataset created, %s items' % (self.real_size))
        self.scale_range = scale_range
        self.src_scale_range = (0.75, 1.5)
        self.resample_range = (0.1, 0.25)
        self.flip_prob = 0.5
        self.ori_prob = ori_prob
        self.max_instance_num = max_instance_num
        self.border_aug_height = 60 if self.class_id == 1 else 100

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def pad_array(self, imgs, pad_size):
        w, h = imgs[0].size
        padding = (0, 0, pad_size[1] - w, pad_size[0] - h)
        pad_mask = np.zeros(pad_size, dtype=np.uint8)   # a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        pad_mask[h:, w:] = 1
        return [TF.pad(el, padding) for el in imgs] + [Image.fromarray(pad_mask)]

    def random_crop(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def getOtherFrames(self, image, instance, label, img_path, isKINS=None):
        if 'image_2' in img_path:
            isKINS = True
        else:
            if random.random() < 0.2:
                isKINS = True
            else:
                isKINS = False
        res = []
        if isKINS:
            scale = random.random() * (self.scale_range[1]-self.scale_range[0]) + self.scale_range[0]
            scale_step = scale * random.random() * (self.resample_range[1]-self.resample_range[0]) + self.resample_range[0]
            h, w = np.array(instance).shape
            if scale < 1.0:
                # small -> big, upsample gradually
                h0, w0 = int(round(h * scale)), int(round(w * scale))
                h1, w1 = int(round(h * (scale+scale_step))), int(round(w * (scale+scale_step)))
                h2, w2 = int(round(h * (scale+scale_step*2))), int(round(w * (scale+scale_step*2)))
            else:
                # big -> small, upsample gradually
                h0, w0 = int(round(h * scale)), int(round(w * scale))
                h1, w1 = int(round(h * (scale - scale_step))), int(round(w * (scale - scale_step)))
                h2, w2 = int(round(h * (scale - scale_step * 2))), int(round(w * (scale - scale_step * 2)))
            for (h_, w_) in [(h0, w0), (h1, w1), (h2, w2)]:
                image_np = cv2.resize(np.array(image), (w_, h_), interpolation=cv2.INTER_CUBIC)
                instance_np = cv2.resize(np.array(instance), (w_, h_), interpolation=cv2.INTER_NEAREST)
                label_np = cv2.resize(np.array(label), (w_, h_), interpolation=cv2.INTER_NEAREST)
                if h_>h: # crop at the center area
                    ch, cw = h_-h, w_-w
                    cy1, cy2 = ch//2, ch//2+h
                    cx1, cx2 = cw//2, cw//2+w
                    image_np, instance_np, label_np = image_np[cy1:cy2, cx1:cx2], instance_np[cy1:cy2, cx1:cx2], label_np[cy1:cy2, cx1:cx2]
                ch, cw = instance_np.shape
                if ch < h or cw < w: # center padding, which is more reasonable
                    ph, pw = h-ch, w-cw
                    ph1, ph2 = ph//2, ph - ph//2
                    pw1, pw2 = pw//2, pw - pw//2
                    image_np = np.pad(image_np,((ph1, ph2), (pw1, pw2), (0,0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np,((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np,((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                res += [image_np, instance_np, label_np]
        else:   # KITTI MOTS real videos
            # find nearby three frames
            prefix, video, frameCount = img_path.rsplit('/', 2)
            frameCount = int(float(frameCount.split('.')[0]))
            frameChoices = [frameCount-3, frameCount-2, frameCount-1, frameCount, frameCount+1, frameCount+2, frameCount+3]
            selFrames = [el for el in frameChoices if 0 <= el < self.TIMESTEPS_PER_SEQ[video]]
            selFrames = sorted(random.sample(selFrames, 3))
            for selFrameCount in selFrames:
                imgPath = os.path.join(prefix, video, '%06d.png' % int(float(selFrameCount)))
                instPath = imgPath.replace('images', 'instances')
                image = Image.open(imgPath)
                instance = Image.open(instPath)
                instance, label = self.decode_instance(instance)
                res += [np.array(el) for el in [image, instance, label]]
        # random Flip
        if random.random() < self.flip_prob:
            res = [el[:, ::-1] for el in res]
        return [Image.fromarray(el) for el in res]

    def cropOutOfScope(self, cropRGB, tseg, y0, x0, y1, x1, hw):
        h, w = hw
        th, tw, _ = cropRGB.shape
        if y0 < 0:
            if -y0 > th:
                raise NotImplementedError
            cropRGB, tseg = cropRGB[-y0:], tseg[-y0:]
            y0 = 0
        if x0 < 0:
            if -x0 > tw:
                raise NotImplementedError
            cropRGB, tseg = cropRGB[:,-x0:], tseg[:,-x0:]
            x0 = 0
        if y1 > h:
            if y1-th>=h:
                raise NotImplementedError
            cropRGB, tseg = cropRGB[:h-y1-1], tseg[:h-y1-1]
            # y1 = h
        if x1 > w:
            if x1-tw>=w:
                raise NotImplementedError
            cropRGB, tseg = cropRGB[:, :w-x1-1], tseg[:, :w-x1-1]
            # x1 = w
        th, tw, _ = cropRGB.shape
        return cropRGB, tseg, y0, x0, y0+th, x0+tw

    def copy_and_paste(self, targets, srcs, next_instance_id):
        res = []
        instIDDict = {}
        instSPDict = {}
        for (image, instance, label), instDict in zip(targets, srcs):
            image_np, instance_np, label_np = np.array(image), np.array(instance), np.array(label)

            if self.paste_by_order: # paste from large to small
                instDict = sorted(instDict, key=lambda k: k['area'], reverse=True)  # from large to small, better
            for inst in instDict:
                try:
                    uid, cropRGB, tseg, xywh, sp = inst['uid'], inst['crop'], inst['seg'], inst['box'], inst['sp']

                    x, y, w0, h0 = xywh
                    if self.non_shift: # non_shift by default
                        sp = inst['sp']
                    elif self.jitter:
                        u_delta = w0 * random.random() * 0.5 * (-1 if random.random()>0.5 else 1)
                        v_delta = h0 * random.random() * 0.25 * (-1 if random.random()>0.5 else 1)
                        v, u = inst['sp']
                        sp = (v+int(round(v_delta)), u+int(round(u_delta)))
                    else:
                        # find a new start point for the instance block
                        if uid in list(instSPDict.keys()):
                            sp = instSPDict[uid]
                        else:
                            locs, overlaps = [], []
                            for _ in range(2):
                                # try 2 times to find a loc
                                i, j, h, w = self.random_crop(image, (h0, w0))
                                y0_, x0_, y1_, x1_ = i, j, i + h, j + w
                                occupy = instance_np[y0_:y1_, x0_:x1_] > 0
                                overlap = (occupy & tseg).sum()
                                locs.append([y0_, x0_])
                                overlaps.append(overlap)
                            best_loc = locs[overlaps.index(min(overlaps))]
                            sp = best_loc

                    v0, u0 = sp
                    y0, x0, y1, x1 = y + v0, x + u0, y + v0 + h0, x + u0 + w0
                    cropRGB, tseg, y0, x0, y1, x1 = self.cropOutOfScope(cropRGB, tseg, y0, x0, y1, x1,instance_np.shape)
                except:
                    pass
                    continue

                tpixels = cropRGB[tseg.astype(np.bool)]
                aug_mask = np.zeros(instance_np.shape, dtype=np.uint8)
                aug_mask[y0:y1, x0:x1] = tseg
                image_np[aug_mask.astype(np.bool)] = tpixels

                if uid in list(instIDDict.keys()):
                    target_id = instIDDict[uid]
                else:
                    instIDDict[uid] = next_instance_id
                    target_id = next_instance_id
                    next_instance_id += 1
                instance_np[aug_mask.astype(np.bool)] = target_id
                label_np[aug_mask.astype(np.bool)] = 1

            res += [image_np, instance_np, label_np]
        uids = np.unique(np.array(np.unique(res[1]).tolist() + np.unique(res[4]).tolist() + np.unique(res[7]).tolist()))
        uids = uids[uids > 0]
        if uids.shape[0] < self.min_inst_num:
            raise NotImplementedError
        return [Image.fromarray(el) for el in res]

    def getSrcs(self, sample_num):
        inst_ids = random.sample(self.instancesKeys, sample_num)

        instDicts = [[] for _ in range(3)]
        non_left, non_right = True, True
        for _, inst_id in enumerate(inst_ids):
            instCrops = self.instancesSegDB[inst_id]
            inst_length = len(instCrops)
            if inst_length > 3:
                mid = random.choice(range(1, inst_length - 2))
                nearby = self.nearby
                start, end = max(0, mid - nearby), min(inst_length - 1, mid + nearby)
                start, end = random.choice(range(start, mid)), random.choice(range(mid + 1, end + 1))
                pis = [instCrops[start], instCrops[mid], instCrops[end]]
            else:
                pis = instCrops[:]
            if random.random() > 0.5: # random replay
                pis = pis[::-1]
            # zero the original loc
            sps = np.array([el['sp'] for el in pis])
            min_v, min_u = sps[:,0].min(), sps[:,1].min()
            max_v, max_u = sps[:,0].max(), sps[:,1].max()
            for p in pis:
                x0, y0, x1, y1 = p['xyxy']
                p['xyxy'] = [x0-min_u, y0-min_v, x1-min_u, y1-min_v]
                # v, u = p['sp']
                p['sp'] = [min_v, min_u]
            if self.border_aug and non_right and pis[0]['xyxy'][3] - pis[0]['xyxy'][1] > self.border_aug_height: # right aug
                img_, mask_ = pis[0]['img'], pis[0]['mask']
                x0, y0, x1, y1 = pis[0]['xyxy']
                w_ = x1 - x0
                xyxy_ = [x0 - x0, y0, x1 - x0, y1]  # zero the position
                seq = sorted([int(1242 - w_ + w_ * (0.2 * random.random() + 0.2)),
                              int(1242 - w_ + w_ * (0.2 * random.random() + 0.4)),
                              int(1242 - w_ + w_ * (0.2 * random.random() + 0.7))])
                for ind, p in enumerate(pis):
                    v_, u_ = p['sp']
                    p['sp'] = [v_, seq[ind]]
                    p['img'], p['mask'], p['xyxy'] = img_, mask_, xyxy_
                non_right = False
            elif self.border_aug and non_left and pis[0]['xyxy'][3]-pis[0]['xyxy'][1]>self.border_aug_height: # left aug
                img_, mask_ = pis[0]['img'], pis[0]['mask']
                x0, y0, x1, y1 = pis[0]['xyxy']
                w_ = x1 - x0
                xyxy_ = [x0 - x0, y0, x1 - x0, y1]  # zero the position
                seq = sorted([int(-w_*(0.2*random.random()+0.2)), int(-w_*(0.2*random.random()+0.4)), int(-w_*(0.2*random.random()+0.7))])
                for ind, p in enumerate(pis):
                    v_, u_ = p['sp']
                    p['sp'] = [v_, seq[ind]] # adjust to outOfScope
                    p['img'], p['mask'], p['xyxy'] = img_, mask_, xyxy_
                non_left = False
            else:
                pass
            if self.randomCropShift and random.random() < 0.2:
                randNum = random.random()
                for p in pis:
                    xyxy, tseg = p['xyxy'], p['mask']
                    tx, ty, tw, th = xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                    tseg, (tx, ty, tw, th) = self.random_crop_shift(tseg, (tx, ty, tw, th), (370, 1240), randNum)
                    p['xyxy'], p['mask'] = [tx,ty,tx+tw,ty+th], tseg
            for ind, inst in enumerate(pis):
                (x0, y0, x1, y1), img, mask, sp = inst['xyxy'], inst['img'], inst['mask'], inst['sp']
                instDicts[ind].append({'uid': inst_id, 'crop': img, 'sp': sp, 'seg': mask, 'box': [x0, y0, x1-x0, y1-y0], 'area': mask.sum(), 'upsample': None, 'scale_step': None})
        return instDicts

    def random_crop_shift(self, tseg, box, shape, randNum):
        tseg_ = tseg.copy()
        x_, y_, w_, h_ = box

        x, y, w, h = box
        H, W = shape
        # random shift
        x_delta = int(round(w * (random.random() * 0.5) * (-1 if random.random()>0.5 else 1)))
        y_delta = int(round(h * (random.random() * 0.3) * (-1 if random.random()>0.5 else 1)))
        x = min(max(0, x+x_delta), W-w-1)
        y = min(max(0, y+y_delta), H-h-1)

        # random crop
        crop_ratio = random.random()*0.45+0.05
        if randNum < 0.25: # crop top
            v_start = int(round(h*crop_ratio))
            tseg[:v_start] = False
        elif randNum < 0.5: # crop bottom
            v_end = int(round(h*(1-crop_ratio)))
            tseg[v_end:] = False
        elif randNum < 0.75: # crop left
            u_start = int(round(w*crop_ratio))
            tseg[:, :u_start] = False
        else:   # crop right
            u_end = int(round(w * (1-crop_ratio)))
            tseg[:, u_end:] = False
        if tseg.sum() > self.min_inst_pixel:
            return tseg, (x,y,w,h)
        else:
            return tseg_, (x_,y_,w_,h_)

    def get_data_from_kins(self, index):
        if not self.type in ['val', 'test']:
            index = random.randint(0, self.real_size - 1)
            # Load the target sample
            sample = {}
            image = Image.open(self.mots_image_list[index])
            sample['im_name'] = self.mots_image_list[index]
            sample['im_shape'] = np.array([image.size])
            # load instances
            instance = Image.open(self.mots_instance_list[index])
            instance, label = self.decode_instance(instance)
            next_instance_id = np.unique(instance).max() + 1
            inst_count = np.unique(instance)[1:].shape[0]

            image, instance, label, image2, instance2, label2, image3, instance3, label3 = \
                self.getOtherFrames(image, instance, label, self.mots_image_list[index])
            targets = [[image, instance, label], [image2, instance2, label2], [image3, instance3, label3]]

            # Load the reference sample
            srcs = self.getSrcs(random.randint(2, max(min(self.max_instance_num//2, 5), self.max_instance_num - inst_count)))

            image, instance, label, image2, instance2, label2, image3, instance3, label3 = self.copy_and_paste(targets, srcs, next_instance_id)

            image, instance, label, pad_mask = self.pad_array([image, instance, label], self.pad_size)
            image2, instance2, label2, pad_mask2 = self.pad_array([image2, instance2, label2], self.pad_size)
            image3, instance3, label3, pad_mask3 = self.pad_array([image3, instance3, label3], self.pad_size)
            sample['image2'] = image2
            sample['instance2'] = instance2
            sample['label2'] = label2
            sample['pad_mask2'] = pad_mask2
            sample['image3'] = image3
            sample['instance3'] = instance3
            sample['label3'] = label3
            sample['pad_mask3'] = pad_mask3
        else: # val, test
            sample = {}
            image = Image.open(self.mots_image_list[index])
            sample['im_name'] = self.mots_image_list[index]
            w, h = image.size
            sample['im_shape'] = np.array([image.size])
            # load instances
            if self.type == 'val':
                instance = Image.open(self.mots_instance_list[index])
                instance, label = self.decode_instance(instance)
            else:
                instance, label = Image.fromarray(np.zeros((h, w), dtype=np.uint8)), \
                                  Image.fromarray(np.zeros((h, w), dtype=np.uint8))
            # if not self.type == 'val':
            image, instance, label, pad_mask = self.pad_array([image, instance, label], self.pad_size)

        sample['image'] = image
        sample['instance'] = instance
        sample['label'] = label
        sample['pad_mask'] = pad_mask
        return sample

    def get_data_from_crops(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        # sample = self.get_data_from_kins(index)
        while 1:
            try:
                sample = self.get_data_from_kins(index)
                break
            except:
                pass
        if (self.transform is not None):
            sample = self.transform(sample)
        return sample

    def decode_instance(self, pic):
        class_id = self.class_id
        pic = np.array(pic, copy=False)

        instance_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.uint8)

        mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
        if mask.sum() > 0:
            instance_map[mask] = (pic[mask] + 1) % 1000
            class_map[mask] = 1

        # assign vehicles but not car to -2, dontcare
        mask = np.logical_and(pic > 0, pic < 1000)
        mask_others = (pic == 10000) | mask
        if mask_others.sum() > 0:
            class_map[mask_others] = -2

        return Image.fromarray(instance_map), Image.fromarray(class_map)



