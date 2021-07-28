import glob
import os
import random
import numpy as np
import math
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.segmentation import relabel_sequential
import torch
from torch.utils.data import Dataset
import cv2
from config import *
from file_utils import *
from utils.mots_util import *
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import pycocotools.mask as maskUtils
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageStat
from collections import Counter


class MOTSCopyAndPasteCars(Dataset):
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
                 add_kins=True, centerRadius=False, scale_range = (0.2, 2.0), ori_prob = 0.2, add_crop=False,
                 crop_only=False, requireObject=False, check_paste_quality=False, center_pad=False, apollo=False):
        print('val: ', type=='val', ',centerRadius: ', centerRadius, ',scale_range: ', scale_range, ',ori_prob: ', ori_prob,
              ',add_crop: ', add_crop, ',crop_only: ', crop_only, ',check_paste_quality: ', check_paste_quality)
        self.class_id = class_id
        self.apollo = apollo # APOLLO rather than kitti
        self.img_size = img_size    # (h,w)
        self.img_sizeT = (img_size[1], img_size[0])    # (w, h)
        self.type = type
        self.pad_size = (384, 1248)
        self.center_pad = center_pad
        # self.crop_size = (1024, 320)
        self.centerRadius = centerRadius # compute center and radius for each instance
        self.add_crop = add_crop
        self.crop_only = crop_only
        self.requireObject = requireObject # require at least one FG object
        self.check_paste_quality = check_paste_quality # check paste object quality
        self.min_pixel = 48 # min paste pixel

        if self.apollo:
            self.mots_instance_root = os.path.join(apolloRoot, 'instances')
            self.mots_image_root = os.path.join(apolloRoot, 'images')
            SEQ_TRAIN = {'road0_Record037_1': 462, 'road2_Record006_0': 43, 'road2_Record034_1': 22,
                         'road2_Record035_2': 114, 'road2_Record027_5': 31, 'road2_Record027_9': 31,
                         'road2_Record032_0': 270, 'road3_Record013_9': 82, 'road3_Record014_5': 52,
                         'road0_Record016_1': 69, 'road0_Record032_0': 68, 'road0_Record068_0': 78,
                         'road0_Record086_2': 141, 'road0_Record103_2': 28, 'road1_Record001_0': 69,
                         'road1_Record002_0': 82, 'road1_Record003_0': 145, 'road1_Record004_2': 71,
                         'road2_Record001_1': 55, 'road2_Record024_3': 23, 'road2_Record027_6': 56,
                         'road2_Record030_1': 75, 'road2_Record031_1': 69, 'road2_Record035_1': 345,
                         'road3_Record001_1': 166, 'road3_Record003_4': 191, 'road3_Record013_14': 102,
                         'road3_Record015_7': 158,
                         'road0_Record015_1': 184, 'road0_Record016_0': 29, 'road0_Record033_2': 55,
                         'road0_Record057_1': 127, 'road0_Record058_0': 86, 'road0_Record058_1': 373,
                         'road0_Record068_1': 169, 'road0_Record085_0': 40, 'road0_Record086_0': 121,
                         'road0_Record102_1': 265, 'road1_Record020_2': 27, 'road2_Record006_4': 180,
                         'road2_Record027_7': 25, 'road2_Record029_1': 183, 'road2_Record035_6': 30,
                         'road3_Record003_3': 38, 'road3_Record013_1': 22, 'road3_Record013_10': 265,
                         'road1_Record004_4': 142, 'road1_Record005_1': 152, 'road1_Record006_0': 121,
                         'road1_Record006_1': 179, 'road1_Record018_0': 118, 'road1_Record019_0': 322,
                         'road2_Record035_4': 88, 'road2_Record027_4': 140, 'road3_Record003_2': 195,
                         }
            SEQ_VAL = {'road0_Record037_2': 64, 'road0_Record013_2': 134, 'road0_Record032_2': 124,
                       'road0_Record088_3': 99, 'road2_Record006_1': 22, 'road3_Record001_0': 89,
                       'road3_Record014_7': 24, 'road3_Record015_8': 64, 'road3_Record013_8': 107,
                       'road0_Record014_0': 311, 'road0_Record104_0': 143, 'road1_Record021_0': 115,
                       'road2_Record002_0': 54, 'road2_Record003_2': 62, 'road2_Record006_2': 222,
                       'road2_Record021_1': 113, 'road3_Record015_0': 185,
                       'road0_Record031_0': 344, 'road0_Record036_2': 253, 'road0_Record103_0': 221,
                       'road1_Record002_2': 105, 'road1_Record021_2': 72, 'road2_Record024_1': 179,
                       'road2_Record027_2': 185, 'road2_Record030_0': 92, 'road3_Record013_4': 174,
                       'road1_Record004_3': 274, 'road1_Record005_0': 214, 'road0_Record067_1': 314,
                       'road3_Record014_2': 355,
                       }
            SEQ_TEST = {'road0_Record012_1': 95, 'road0_Record012_2': 192, 'road0_Record033_0': 130,
                        'road0_Record054_0': 77, 'road0_Record054_1': 60, 'road0_Record055_1': 73,
                        'road0_Record070_1': 108, 'road0_Record087_4': 109, 'road0_Record088_0': 131,
                        'road2_Record002_3': 32, 'road2_Record025_0': 66, 'road2_Record025_1': 46,
                        'road2_Record027_8': 63, 'road3_Record001_3': 37, 'road3_Record001_4': 77,
                        'road3_Record002_0': 30, 'road3_Record002_1': 60, 'road3_Record013_6': 40,
                        'road3_Record014_0': 72, 'road3_Record014_4': 104, 'road3_Record015_3': 62,
                        'road3_Record015_4': 22, 'road3_Record015_5': 23, 'road3_Record015_6': 65,
                        'road0_Record011_0': 353, 'road0_Record015_0': 74, 'road0_Record037_0': 171,
                        'road0_Record054_2': 75, 'road0_Record070_2': 32, 'road0_Record087_0': 167,
                        'road0_Record088_4': 54, 'road0_Record105_1': 230, 'road1_Record002_1': 81,
                        'road1_Record004_0': 53, 'road1_Record004_1': 44, 'road1_Record021_1': 133,
                        'road2_Record001_2': 50, 'road2_Record003_3': 71, 'road2_Record003_5': 139,
                        'road2_Record004_1': 47, 'road2_Record006_5': 52, 'road2_Record021_3': 101,
                        'road2_Record022_2': 144, 'road2_Record031_0': 193, 'road2_Record035_3': 118,
                        'road3_Record003_1': 84, 'road3_Record005_1': 171, 'road3_Record013_15': 82,
                        'road3_Record013_17': 141, 'road3_Record013_3': 38, 'road3_Record014_1': 82,
                        'road0_Record032_1': 452, 'road0_Record033_1': 136, 'road0_Record036_1': 353,
                        'road0_Record055_0': 360, 'road0_Record057_0': 183, 'road0_Record067_0': 279,
                        'road0_Record069_1': 406, 'road1_Record003_1': 69, 'road1_Record020_1': 281,
                        'road2_Record003_4': 110, 'road2_Record004_0': 87, 'road2_Record022_1': 58,
                        'road2_Record024_2': 151, 'road2_Record027_3': 158, 'road2_Record035_5': 45,
                        'road3_Record001_2': 115, 'road3_Record002_3': 195, 'road3_Record004_0': 205,
                        'road3_Record013_0': 22, 'road3_Record013_2': 39, 'road3_Record013_7': 37,
                        'road3_Record014_6': 62,
                        'road0_Record086_1': 438, 'road0_Record103_1': 440, 'road1_Record018_1': 22,
                        'road1_Record019_1': 30, 'road1_Record020_0': 207, 'road2_Record006_3': 84,
                        'road2_Record022_0': 192, 'road2_Record024_4': 243, 'road3_Record003_0': 99,
                        'road3_Record013_12': 348, 'road3_Record014_3': 332,
                        }
            # TIMESTEPS_PER_SEQ = SEQ_TRAIN
            # TIMESTEPS_PER_SEQ.update(SEQ_VAL)
            SEQ_IDS_TRAIN = list(SEQ_TRAIN.keys())
            SEQ_IDS_VAL = list(SEQ_VAL.keys())
            SEQ_IDS_TEST = list(SEQ_TEST.keys())
            if 'train' in type:
                self.SEQs = SEQ_TRAIN
                self.keys = SEQ_IDS_TRAIN
            else: # val
                self.SEQs = SEQ_VAL
                self.keys = SEQ_IDS_VAL
            print('Total %s images' % sum(self.SEQs.values()))
            self.mots_instance_list, self.mots_image_list = [], []
            for id in self.keys:
                subdir = os.path.join(self.mots_instance_root, id)
                self.mots_instance_list += make_dataset(subdir, suffix='.png')
            self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]

        else: # kitti
            self.mots_instance_root = os.path.join(kittiRoot, 'instances')
            self.mots_image_root = os.path.join(kittiRoot, 'images')
            if type == 'trainval':
                self.squence = self.SEQ_IDS_TRAIN + self.SEQ_IDS_VAL
                mots_persons = load_pickle(os.path.join(kittiRoot, 'mots_inst_train5.pkl'))
                print("train with training and val set")
            elif type == 'test':
                self.squence = self.SEQ_IDS_TEST
                self.mots_image_root = os.path.join(kittiRoot, 'testing/image_02')
                mots_persons = []
                for subdir in self.squence:
                    image_list = sorted(make_dataset(os.path.join(self.mots_image_root, subdir), suffix='.png'))
                    # image_list = ['/'.join(el.split('/')[-2:]) for el in image_list]
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

        self.real_size = len(self.mots_image_list)
        self.size = size
        self.transform = transform

        if self.add_crop and (not type == 'val'):
            self.crop_image_list = make_dataset(os.path.join(kittiRoot.replace('kitti', 'kitti_raw'), 'images'), suffix='.png')
            self.crop_instance_list = [el.replace('images', 'instances') for el in self.crop_image_list]
            print('KittiRaw, %s items' % len(self.crop_instance_list))

        from datasets.ar_transforms.ap_transforms import get_default_ap_transforms
        self.ap_transform = get_default_ap_transforms()
        print('KittiMOTS Dataset created, %s items' % (self.real_size))
        self.scale_range = scale_range
        self.flip_prob = 0.5
        self.ori_prob = ori_prob
        self.max_instance_num = 25 # evade OOM, too many instances

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

    def randScaleAndFlip(self, image, instance, label):
        scale = random.random() * (self.scale_range[1]-self.scale_range[0]) + self.scale_range[0]
        image_np, instance_np, label_np = np.array(image), np.array(instance), np.array(label)
        h, w = instance_np.shape
        h_, w_ = int(round(h*scale)), int(round(w*scale))
        image_np = cv2.resize(image_np, (w_, h_), interpolation=cv2.INTER_CUBIC)
        instance_np = cv2.resize(instance_np, (w_, h_), interpolation=cv2.INTER_NEAREST)
        label_np = cv2.resize(label_np, (w_, h_), interpolation=cv2.INTER_NEAREST)
        if h_>h:
            if self.center_pad:# crop center area
                ch, cw = h_-h, w_-w
                cy1, cy2 = ch//2, ch//2+h
                cx1, cx2 = cw//2, cw//2+w
                image_np, instance_np, label_np = image_np[cy1:cy2, cx1:cx2], instance_np[cy1:cy2, cx1:cx2], label_np[cy1:cy2, cx1:cx2]
            else:
                image_np, instance_np, label_np = image_np[:h, :w], instance_np[:h, :w], label_np[:h, :w]
        ch, cw = instance_np.shape
        if ch<h or cw<w:
            if self.center_pad:
                ph, pw = h - ch, w - cw
                ph1, ph2 = ph // 2, ph - ph // 2
                pw1, pw2 = pw // 2, pw - pw // 2
                try:
                    image_np = np.pad(image_np, ((ph1, ph2), (pw1, pw2), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                except:
                    image_np = np.pad(image_np, ((0, h - ch), (0, w - cw), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
            else:
                # random pad
                randNum = random.random()
                if randNum < 0.33:  # left
                    image_np = np.pad(image_np, ((0, h - ch), (0, w - cw), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
                elif 0.33 <= randNum <= 0.67:
                    ph, pw = h - ch, w - cw
                    ph1, ph2 = ph // 2, ph - ph // 2
                    pw1, pw2 = pw // 2, pw - pw // 2
                    image_np = np.pad(image_np, ((ph1, ph2), (pw1, pw2), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                else:  # right
                    image_np = np.pad(image_np, ((h - ch, 0), (w - cw, 0), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((h - ch, 0), (w - cw, 0)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((h - ch, 0), (w - cw, 0)), mode='constant', constant_values=0)

        # random Flip
        if random.random() < self.flip_prob:
            image_np, instance_np, label_np = image_np[:, ::-1], instance_np[:, ::-1], label_np[:, ::-1]

        # cv2.imwrite('/data/xuzhenbo/1.jpg', image_np[:,:,::-1])
        # cv2.imwrite('/data/xuzhenbo/2.jpg', instance_np*50)
        return Image.fromarray(image_np), Image.fromarray(instance_np), Image.fromarray(label_np)

    def copy_and_paste(self, image, instance, label, image2, instance2, label2):
        image_np, image2_np = np.array(image), np.array(image2)
        instance_np, instance2_np = np.array(instance), np.array(instance2)
        label_np, label2_np = np.array(label), np.array(label2)
        ids1 = np.unique(instance_np)[1:] # reserverd for removing invisible objects

        id_index = instance_np.max() + 1
        ids2 = np.unique(instance2_np)[1:]
        if len(ids2) > 0:
            for uid in ids2:
                if id_index > self.max_instance_num:
                    break
                try:
                    # prepare this instance
                    inst_mask = instance2_np == uid
                    coco_seg = maskUtils.encode(np.asfortranarray(inst_mask.astype(np.uint8)))
                    tbbox = maskUtils.toBbox(coco_seg)
                    tx, ty, tw, th = [int(el) for el in tbbox]
                    # if self.check_paste_quality:
                    #     if inst_mask.sum() < self.min_pixel or (not (0.2 < tw/float(th) < 5.0)):
                    #         continue

                    tseg = inst_mask[ty:ty + th, tx:tx + tw]
                    timgRGB = image2_np[ty:ty + th, tx:tx + tw]
                    h_, w_ = tseg.shape
                    tpixels = timgRGB[tseg.astype(np.bool)]

                    # randomly find a position
                    i, j, h, w = self.random_crop(image, (h_, w_))  # (i, j, h, w)
                    y0, x0, y1, x1 = i, j, i + h, j + w
                    aug_mask = np.zeros(instance_np.shape, dtype=np.uint8)
                    aug_mask[y0:y1, x0:x1] = tseg
                    image_np[aug_mask.astype(np.bool)] = tpixels
                    instance_np[aug_mask.astype(np.bool)] = id_index
                    label_np[aug_mask.astype(np.bool)] = 1
                    id_index += 1
                except:
                    continue
            # removing invisible objects
            if self.check_paste_quality:
                for uid in ids1:
                    inst_mask = instance_np == uid
                    if inst_mask.sum() < self.min_pixel: # too small to be considered
                        instance_np[inst_mask] = 0
                        label_np[inst_mask] = -2
            else:
                for uid in ids1:
                    inst_mask = instance_np == uid
                    if inst_mask.sum() < 20:  # too small to be considered
                        instance_np[inst_mask] = 0
                        label_np[inst_mask] = -2

        # cv2.imwrite('/data/xuzhenbo/1.jpg', image_np[:,:,::-1])
        # cv2.imwrite('/data/xuzhenbo/2.jpg', instance_np*50)
        return Image.fromarray(image_np), Image.fromarray(instance_np), Image.fromarray(label_np)

    def getCenterRadius(self, instance):
        instance_np = np.array(instance)
        ids = np.unique(instance_np)[1:]
        centers, radius, uids = [], [], []
        vMax, uMax = self.pad_size
        for uid in ids:
            inst_mask = instance_np == uid
            vs_, us_ = np.nonzero(inst_mask)
            vc, uc = vs_.mean(), us_.mean()
            vr, ur = np.abs(vs_-vc).max(), np.abs(us_-uc).max()
            centers.append([vc/vMax, uc/uMax])
            radius.append([vr/vMax + 1e-5, ur/uMax + 1e-5])
            uids.append(uid)
        return centers, radius, uids

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

            # Load the reference sample
            index2 = random.randint(0, self.real_size - 1)
            image2 = Image.open(self.mots_image_list[index2])
            instance2 = Image.open(self.mots_instance_list[index2])
            instance2, label2 = self.decode_instance(instance2)

            if random.random() > self.ori_prob:
                image, instance, label = self.randScaleAndFlip(image, instance, label)
                image2, instance2, label2 = self.randScaleAndFlip(image2, instance2, label2)
                image, instance, label = self.copy_and_paste(image, instance, label, image2, instance2, label2)
            image, instance, label, pad_mask = self.pad_array([image, instance, label], self.pad_size)
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
        if self.centerRadius:
            centers, radius, ids = self.getCenterRadius(instance)
            sample['center'] = np.array(centers, dtype=np.float32)
            sample['radius'] = np.array(radius, dtype=np.float32)
            sample['ids'] = np.array(ids, dtype=np.float32)
        return sample

    def get_data_from_crops(self, index):
        index = random.randint(0, len(self.crop_image_list) - 1)
        sample = {}
        image = Image.open(self.crop_image_list[index])
        sample['im_name'] = self.crop_image_list[index]
        sample['im_shape'] = np.array([image.size])
        # load instances
        instance = Image.open(self.crop_instance_list[index])
        instance, label = self.decode_instance(instance)

        # Load the reference sample
        index2 = random.randint(0, self.real_size - 1)
        image2 = Image.open(self.mots_image_list[index2])
        instance2 = Image.open(self.mots_instance_list[index2])
        instance2, label2 = self.decode_instance(instance2)

        image, instance, label = self.randScaleAndFlip(image, instance, label)
        image2, instance2, label2 = self.randScaleAndFlip(image2, instance2, label2)
        image, instance, label = self.copy_and_paste(image, instance, label, image2, instance2, label2)
        image, instance, label, pad_mask = self.pad_array([image, instance, label], self.pad_size)

        sample['image'] = image
        sample['instance'] = instance
        sample['label'] = label
        sample['pad_mask'] = pad_mask
        if self.centerRadius:
            centers, radius, ids = self.getCenterRadius(instance)
            sample['center'] = np.array(centers, dtype=np.float32)
            sample['radius'] = np.array(radius, dtype=np.float32)
            sample['ids'] = np.array(ids, dtype=np.float32)
        return sample

    def __getitem__(self, index):
        if not self.requireObject:
            if (not self.add_crop) or (self.type in ['test', 'val']):
                sample = self.get_data_from_kins(index)
            elif self.add_crop and self.crop_only:
                sample = self.get_data_from_crops(index)
            else:
                sample = self.get_data_from_kins(index) if random.random() > 0.5 else self.get_data_from_crops(index)
                # sample = self.get_data_from_crops(index)
        else:
            while 1:
                if (not self.add_crop) or (self.type in ['test', 'val']):
                    sample = self.get_data_from_kins(index)
                elif self.add_crop and self.crop_only:
                    sample = self.get_data_from_crops(index)
                else:
                    sample = self.get_data_from_kins(index) if random.random() > 0.5 else self.get_data_from_crops(index)
                if sample['ids'].shape[0] > 0:
                    break
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
        # if self.type=='crop':
        #     assert mask.sum() > 0
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids
            class_map[mask] = 1

        # assign vehicles but not car to -2, dontcare
        mask = np.logical_and(pic > 0, pic < 1000)
        mask_others = (pic == 10000) | mask
        if mask_others.sum() > 0:
            class_map[mask_others] = -2

        return Image.fromarray(instance_map), Image.fromarray(class_map)


class MOTSCopyAndPasteCarsV2(Dataset):
    # 2021/02/05
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
                 add_kins=True, centerRadius=False, scale_range = (0.2, 2.0), ori_prob = 0.2, add_crop=False,
                 crop_only=False, requireObject=False, check_paste_quality=False, center_pad=False, paste_num=25, add_kins_template=False, min_add_num=0):
        print('val: ', type=='val', ',centerRadius: ', centerRadius, ',scale_range: ', scale_range, ',ori_prob: ', ori_prob,
              ',add_crop: ', add_crop, ',crop_only: ', crop_only, ',check_paste_quality: ', check_paste_quality)
        self.class_id = class_id
        self.img_size = img_size    # (h,w)
        self.img_sizeT = (img_size[1], img_size[0])    # (w, h)
        self.type = type
        self.pad_size = (384, 1248)
        self.center_pad = center_pad
        self.centerRadius = centerRadius # compute center and radius for each instance
        self.add_crop = add_crop
        self.crop_only = crop_only
        self.requireObject = requireObject # require at least one FG object
        self.check_paste_quality = check_paste_quality # check paste object quality
        self.min_pixel = 48 # min paste pixel
        self.paste_num = paste_num
        self.add_kins = add_kins
        self.add_kins_template = add_kins_template # only add template

        self.mots_instance_root = os.path.join(kittiRoot, 'instances')
        self.mots_image_root = os.path.join(kittiRoot, 'images')
        if type == 'trainval':
            # self.squence = self.SEQ_IDS_TRAIN + self.SEQ_IDS_VAL
            mots_persons = load_pickle(os.path.join(kittiRoot, 'mots_inst_train5.pkl'))
            print("train with training and val set")
        elif type == 'test':
            self.squence = self.SEQ_IDS_TEST
            self.mots_image_root = os.path.join(kittiRoot, 'testing/image_02')
            mots_persons = []
            for subdir in self.squence:
                image_list = sorted(make_dataset(os.path.join(self.mots_image_root, subdir), suffix='.png'))
                # image_list = ['/'.join(el.split('/')[-2:]) for el in image_list]
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

        self.kitti_size = len(self.mots_image_list)
        if (add_kins or add_kins_template) and (not type in ['test', 'val']):
            dst_kins = 'person_KINS' if self.class_id == 2 else 'KINS'
            instance_list = make_dataset(os.path.join(kittiRoot, 'training/' + dst_kins), suffix='.png') + make_dataset(
                os.path.join(kittiRoot, 'testing/' + dst_kins), suffix='.png')
            image_list = [el.replace(dst_kins, 'image_2') for el in instance_list]
            print('add KINS: ', len(image_list))
            self.mots_image_list += image_list
            self.mots_instance_list += instance_list
        else:
            print('wo KINS! for finetune')

        self.real_size = len(self.mots_image_list)
        self.size = size
        self.transform = transform

        # Load segs
        if 'train' in type:
            if self.class_id == 1:
                self.KINS_segs = load_pickle(os.path.join(kittiRoot, 'segDB_car_KINS_0205.pkl'))
                self.KITTIMOTS_segs = load_pickle(os.path.join(kittiRoot, 'segDB_car_trainval_0205.pkl' if self.type=='trainval' else 'segDB_car_trainset_0205.pkl'))
            else:
                self.KINS_segs = load_pickle(os.path.join(kittiRoot, 'segDB_person_KINS_0205.pkl'))
                self.KITTIMOTS_segs = load_pickle(os.path.join(kittiRoot, 'segDB_person_trainval_0205.pkl' if self.type=='trainval' else 'segDB_person_trainset_0205.pkl'))
        else:
            self.KINS_segs, self.KITTIMOTS_segs = None, None

        from datasets.ar_transforms.ap_transforms import get_default_ap_transforms
        self.ap_transform = get_default_ap_transforms()
        print('MOTSCopyAndPasteCarsV2 Dataset created, %s items' % (self.real_size))
        self.scale_range = scale_range
        self.flip_prob = 0.5
        self.ori_prob = ori_prob
        self.max_instance_num = 25 # evade OOM, too many instances
        self.min_add_num = min_add_num

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

    def randScaleAndFlip(self, image, instance, label):
        scale = random.random() * (self.scale_range[1]-self.scale_range[0]) + self.scale_range[0]
        image_np, instance_np, label_np = np.array(image), np.array(instance), np.array(label)
        h, w = instance_np.shape
        h_, w_ = int(round(h*scale)), int(round(w*scale))
        image_np = cv2.resize(image_np, (w_, h_), interpolation=cv2.INTER_CUBIC)
        instance_np = cv2.resize(instance_np, (w_, h_), interpolation=cv2.INTER_NEAREST)
        label_np = cv2.resize(label_np, (w_, h_), interpolation=cv2.INTER_NEAREST)
        if h_>h:
            if self.center_pad:# crop center area
                ch, cw = h_-h, w_-w
                cy1, cy2 = ch//2, ch//2+h
                cx1, cx2 = cw//2, cw//2+w
                image_np, instance_np, label_np = image_np[cy1:cy2, cx1:cx2], instance_np[cy1:cy2, cx1:cx2], label_np[cy1:cy2, cx1:cx2]
            else:
                image_np, instance_np, label_np = image_np[:h, :w], instance_np[:h, :w], label_np[:h, :w]
        ch, cw = instance_np.shape
        if ch<h or cw<w:
            if self.center_pad:
                ph, pw = h - ch, w - cw
                ph1, ph2 = ph // 2, ph - ph // 2
                pw1, pw2 = pw // 2, pw - pw // 2
                try:
                    image_np = np.pad(image_np, ((ph1, ph2), (pw1, pw2), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                except:
                    image_np = np.pad(image_np, ((0, h - ch), (0, w - cw), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
            else:
                image_np = np.pad(image_np,((0, h-ch), (0, w-cw), (0,0)), mode='constant', constant_values=0)
                instance_np = np.pad(instance_np,((0, h-ch), (0, w-cw)), mode='constant', constant_values=0)
                label_np = np.pad(label_np,((0, h-ch), (0, w-cw)), mode='constant', constant_values=0)

        # random Flip
        if random.random() < self.flip_prob:
            image_np, instance_np, label_np = image_np[:, ::-1], instance_np[:, ::-1], label_np[:, ::-1]

        # cv2.imwrite('/data/xuzhenbo/1.jpg', image_np[:,:,::-1])
        # cv2.imwrite('/data/xuzhenbo/2.jpg', instance_np*50)
        return Image.fromarray(image_np), Image.fromarray(instance_np), Image.fromarray(label_np)

    def copy_and_paste(self, image, instance, label, instDicts):
        image_np, instance_np, label_np = np.array(image), np.array(instance), np.array(label)
        ids1 = np.unique(instance_np)[1:] # reserverd for removing invisible objects

        id_index = instance_np.max() + 1
        ids2 = []
        for inst_info in instDicts:

            # prepare this instance
            tx, ty, tw, th = inst_info['box']
            tseg, timgRGB = inst_info['seg'], inst_info['crop']

            # random scale
            if 12<th<125:
                scale = random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            elif th<=12:
                scale_range = (1.0, self.scale_range[1])
                scale = random.random() * (scale_range[1] - scale_range[0]) + scale_range[0]
            else:
                scale_range = (self.scale_range[0], 1.5)
                scale = random.random() * (scale_range[1] - scale_range[0]) + scale_range[0]

            scale = min(300/th, scale) # max_height 300
            try:
                h_, w_ = int(round(th * scale)), int(round(tw * scale))
                timgRGB = cv2.resize(timgRGB, (w_, h_), interpolation=cv2.INTER_CUBIC)
                tseg = cv2.resize(tseg.astype(np.uint8), (w_, h_), interpolation=cv2.INTER_NEAREST)

                # random Flip
                if random.random() < self.flip_prob:
                    timgRGB, tseg = timgRGB[:, ::-1], tseg[:, ::-1]

                tpixels = timgRGB[tseg.astype(np.bool)]
                # randomly find a position
                h_, w_ = tseg.shape
                i, j, h, w = self.random_crop(image, (h_, w_))  # (i, j, h, w)
                y0, x0, y1, x1 = i, j, i + h, j + w
                aug_mask = np.zeros(instance_np.shape, dtype=np.uint8)
                aug_mask[y0:y1, x0:x1] = tseg
                image_np[aug_mask.astype(np.bool)] = tpixels
                instance_np[aug_mask.astype(np.bool)] = id_index
                label_np[aug_mask.astype(np.bool)] = 1
                ids2.append(id_index)
                id_index += 1
            except:
                print(image.size, h_, w_)
                continue
        # removing invisible objects
        for uid in ids1.tolist()+ids2:
            inst_mask = instance_np == uid
            if inst_mask.sum() < self.min_pixel:  # too small to be considered
                instance_np[inst_mask] = 0
                label_np[inst_mask] = -2

        # cv2.imwrite('/data/xuzhenbo/1.jpg', image_np[:,:,::-1])
        # cv2.imwrite('/data/xuzhenbo/2.jpg', instance_np*50)
        return Image.fromarray(image_np), Image.fromarray(instance_np), Image.fromarray(label_np)

    def getCenterRadius(self, instance):
        instance_np = np.array(instance)
        ids = np.unique(instance_np)[1:]
        centers, radius, uids = [], [], []
        vMax, uMax = self.pad_size
        for uid in ids:
            inst_mask = instance_np == uid
            vs_, us_ = np.nonzero(inst_mask)
            vc, uc = vs_.mean(), us_.mean()
            vr, ur = np.abs(vs_-vc).max(), np.abs(us_-uc).max()
            centers.append([vc/vMax, uc/uMax])
            radius.append([vr/vMax + 1e-5, ur/uMax + 1e-5])
            uids.append(uid)
        return centers, radius, uids

    def getSrcs(self, sample_num):
        if self.add_kins:
            kins_insts = random.sample(self.KINS_segs, sample_num // 2)
            KITTIMOTS_insts = random.sample(self.KITTIMOTS_segs, sample_num // 2)
        else:
            kins_insts = []
            KITTIMOTS_insts = random.sample(self.KITTIMOTS_segs, sample_num)

        instDicts = []
        for ind, inst in enumerate(kins_insts+KITTIMOTS_insts):
            # random flip and rescale
            (x0, y0, x1, y1), img, mask, sp = inst['xyxy'], inst['img'], inst['mask'], inst['sp']
            instDicts.append({'crop': img, 'sp': sp, 'seg': mask, 'box': [x0, y0, x1-x0, y1-y0],'upsample': None, 'scale_step': None})
        return instDicts

    def get_data_from_kins(self, index):
        if not self.type in ['val', 'test']:
            if self.add_kins_template:
                if random.random() < 0.5:
                    index = random.randint(0, self.kitti_size - 1)
                else:
                    index = random.randint(self.kitti_size, self.real_size - 1)
            else:
                index = random.randint(0, self.real_size - 1)
            # Load the target sample
            sample = {}
            image = Image.open(self.mots_image_list[index])
            sample['im_name'] = self.mots_image_list[index]
            sample['im_shape'] = np.array([image.size])
            # load instances
            instance = Image.open(self.mots_instance_list[index])
            instance, label = self.decode_instance(instance)
            instance_count = np.unique(instance).shape[0] - 1

            if random.random() > self.ori_prob:
                image, instance, label = self.randScaleAndFlip(image, instance, label)
                instDicts = self.getSrcs(random.randint(self.min_add_num, max(self.paste_num-instance_count, self.min_add_num+1))) # random get insts with a random number
                image, instance, label = self.copy_and_paste(image, instance, label, instDicts)
            image, instance, label, pad_mask = self.pad_array([image, instance, label], self.pad_size)
            sample['seed_w'] = [0.3 if 'KINS' in self.mots_instance_list[index] else 1.0]
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
            sample['seed_w'] = [1.0]

        sample['image'] = image
        sample['instance'] = instance
        sample['label'] = label
        sample['pad_mask'] = pad_mask
        if self.centerRadius:
            centers, radius, ids = self.getCenterRadius(instance)
            sample['center'] = np.array(centers, dtype=np.float32)
            sample['radius'] = np.array(radius, dtype=np.float32)
            sample['ids'] = np.array(ids, dtype=np.float32)
        return sample

    def __getitem__(self, index):
        if not self.requireObject:
            sample = self.get_data_from_kins(index)
        else:
            while 1:
                sample = self.get_data_from_kins(index)
                if sample['ids'].shape[0] > 0:
                    break
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
        # if self.type=='crop':
        #     assert mask.sum() > 0
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids
            class_map[mask] = 1

        # assign vehicles but not car to -2, dontcare
        mask = np.logical_and(pic > 0, pic < 1000)
        mask_others = (pic == 10000) | mask
        if mask_others.sum() > 0:
            class_map[mask_others] = -2

        return Image.fromarray(instance_map), Image.fromarray(class_map)


class MOTSCopyAndPasteCarsV3(Dataset):
    # 0208 similar to the first version, but paste two frames
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
                 add_kins=True, centerRadius=False, scale_range = (0.2, 2.0), ori_prob = 0.2, add_crop=False,
                 crop_only=False, requireObject=False, check_paste_quality=False, center_pad=False):
        print('val: ', type=='val', ',centerRadius: ', centerRadius, ',scale_range: ', scale_range, ',ori_prob: ', ori_prob,
              ',add_crop: ', add_crop, ',crop_only: ', crop_only, ',check_paste_quality: ', check_paste_quality)
        self.class_id = class_id
        self.img_size = img_size    # (h,w)
        self.img_sizeT = (img_size[1], img_size[0])    # (w, h)
        self.type = type
        self.pad_size = (384, 1248)
        self.center_pad = center_pad
        # self.crop_size = (1024, 320)
        self.centerRadius = centerRadius # compute center and radius for each instance
        self.add_crop = add_crop
        self.crop_only = crop_only
        self.requireObject = requireObject # require at least one FG object
        self.check_paste_quality = check_paste_quality # check paste object quality
        self.min_pixel = 48 # min paste pixel

        self.mots_instance_root = os.path.join(kittiRoot, 'instances')
        self.mots_image_root = os.path.join(kittiRoot, 'images')
        if type == 'trainval':
            self.squence = self.SEQ_IDS_TRAIN + self.SEQ_IDS_VAL
            mots_persons = load_pickle(os.path.join(kittiRoot, 'mots_inst_train5.pkl'))
            print("train with training and val set")
        elif type == 'test':
            self.squence = self.SEQ_IDS_TEST
            self.mots_image_root = os.path.join(kittiRoot, 'testing/image_02')
            mots_persons = []
            for subdir in self.squence:
                image_list = sorted(make_dataset(os.path.join(self.mots_image_root, subdir), suffix='.png'))
                # image_list = ['/'.join(el.split('/')[-2:]) for el in image_list]
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

        self.real_size = len(self.mots_image_list)
        self.size = size
        self.transform = transform

        if self.add_crop and (not type == 'val'):
            self.crop_image_list = make_dataset(os.path.join(kittiRoot.replace('kitti', 'kitti_raw'), 'images'), suffix='.png')
            self.crop_instance_list = [el.replace('images', 'instances') for el in self.crop_image_list]
            print('KittiRaw, %s items' % len(self.crop_instance_list))

        from datasets.ar_transforms.ap_transforms import get_default_ap_transforms
        self.ap_transform = get_default_ap_transforms()
        print('MOTSCopyAndPasteCarsV3 Dataset created, %s items' % (self.real_size))
        self.scale_range = scale_range
        self.flip_prob = 0.5
        self.ori_prob = ori_prob
        self.max_instance_num = 25 # evade OOM, too many instances

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

    def randScaleAndFlip(self, image, instance, label):
        scale = random.random() * (self.scale_range[1]-self.scale_range[0]) + self.scale_range[0]
        image_np, instance_np, label_np = np.array(image), np.array(instance), np.array(label)
        h, w = instance_np.shape
        h_, w_ = int(round(h*scale)), int(round(w*scale))
        image_np = cv2.resize(image_np, (w_, h_), interpolation=cv2.INTER_CUBIC)
        instance_np = cv2.resize(instance_np, (w_, h_), interpolation=cv2.INTER_NEAREST)
        label_np = cv2.resize(label_np, (w_, h_), interpolation=cv2.INTER_NEAREST)
        if h_>h:
            if self.center_pad:# crop center area
                ch, cw = h_-h, w_-w
                cy1, cy2 = ch//2, ch//2+h
                cx1, cx2 = cw//2, cw//2+w
                image_np, instance_np, label_np = image_np[cy1:cy2, cx1:cx2], instance_np[cy1:cy2, cx1:cx2], label_np[cy1:cy2, cx1:cx2]
            else:
                image_np, instance_np, label_np = image_np[:h, :w], instance_np[:h, :w], label_np[:h, :w]
        ch, cw = instance_np.shape
        if ch<h or cw<w:
            if self.center_pad:
                ph, pw = h - ch, w - cw
                ph1, ph2 = ph // 2, ph - ph // 2
                pw1, pw2 = pw // 2, pw - pw // 2
                try:
                    # if ph1<=0 or ph2<0 or pw1<0 or pw2<0:
                    # print(scale, image_np.shape, ph1, ph2, pw1, pw2, w, h, ch, cw)
                    image_np = np.pad(image_np, ((ph1, ph2), (pw1, pw2), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((ph1, ph2), (pw1, pw2)), mode='constant', constant_values=0)
                except:
                    # if h - ch<0 or w-cw<0:
                    #     print(scale, image_np.shape, ph1, ph2, pw1, pw2, w, h, ch, cw)
                    #     print(scale, image_np.shape, ph1, ph2, pw1, pw2, w, h, ch, cw)
                    #     print(scale, image_np.shape, ph1, ph2, pw1, pw2, w, h, ch, cw)
                    #     print(scale, image_np.shape, ph1, ph2, pw1, pw2, w, h, ch, cw)
                    image_np = np.pad(image_np, ((0, h - ch), (0, w - cw), (0, 0)), mode='constant', constant_values=0)
                    instance_np = np.pad(instance_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
                    label_np = np.pad(label_np, ((0, h - ch), (0, w - cw)), mode='constant', constant_values=0)
            else:
                image_np = np.pad(image_np,((0, h-ch), (0, w-cw), (0,0)), mode='constant', constant_values=0)
                instance_np = np.pad(instance_np,((0, h-ch), (0, w-cw)), mode='constant', constant_values=0)
                label_np = np.pad(label_np,((0, h-ch), (0, w-cw)), mode='constant', constant_values=0)

        # random Flip
        if random.random() < self.flip_prob:
            image_np, instance_np, label_np = image_np[:, ::-1], instance_np[:, ::-1], label_np[:, ::-1]

        # cv2.imwrite('/data/xuzhenbo/1.jpg', image_np[:,:,::-1])
        # cv2.imwrite('/data/xuzhenbo/2.jpg', instance_np*50)
        return Image.fromarray(image_np), Image.fromarray(instance_np), Image.fromarray(label_np)

    def copy_and_paste(self, image, instance, label, image2, instance2, label2):
        image_np, image2_np = np.array(image), np.array(image2)
        instance_np, instance2_np = np.array(instance), np.array(instance2)
        label_np, label2_np = np.array(label), np.array(label2)
        ids1 = np.unique(instance_np)[1:].tolist() # reserverd for removing invisible objects

        id_index = instance_np.max() + 1
        ids2 = np.unique(instance2_np)[1:]
        new_ids = []
        if len(ids2) > 0:
            for uid in ids2:
                if id_index > self.max_instance_num:
                    break
                try:
                    # prepare this instance
                    inst_mask = instance2_np == uid
                    coco_seg = maskUtils.encode(np.asfortranarray(inst_mask.astype(np.uint8)))
                    tbbox = maskUtils.toBbox(coco_seg)
                    tx, ty, tw, th = [int(el) for el in tbbox]
                    if self.check_paste_quality:
                        if inst_mask.sum() < self.min_pixel or (not (0.2 < tw/float(th) < 5.0)):
                            continue

                    tseg = inst_mask[ty:ty + th, tx:tx + tw]
                    timgRGB = image2_np[ty:ty + th, tx:tx + tw]
                    h_, w_ = tseg.shape
                    tpixels = timgRGB[tseg.astype(np.bool)]

                    # randomly find a position
                    i, j, h, w = self.random_crop(image, (h_, w_))  # (i, j, h, w)
                    y0, x0, y1, x1 = i, j, i + h, j + w
                    aug_mask = np.zeros(instance_np.shape, dtype=np.uint8)
                    aug_mask[y0:y1, x0:x1] = tseg
                    image_np[aug_mask.astype(np.bool)] = tpixels
                    instance_np[aug_mask.astype(np.bool)] = id_index
                    label_np[aug_mask.astype(np.bool)] = 1
                    new_ids.append(id_index)
                    id_index += 1
                except:
                    continue
            # removing invisible objects
            if self.check_paste_quality:
                for uid in ids1+new_ids:
                    inst_mask = instance_np == uid
                    if inst_mask.sum() < self.min_pixel: # too small to be considered
                        instance_np[inst_mask] = 0
                        label_np[inst_mask] = -2
            else:
                for uid in ids1+new_ids:
                    inst_mask = instance_np == uid
                    if inst_mask.sum() < 20:  # too small to be considered
                        instance_np[inst_mask] = 0
                        label_np[inst_mask] = -2

        # cv2.imwrite('/data/xuzhenbo/1.jpg', image_np[:,:,::-1])
        # cv2.imwrite('/data/xuzhenbo/2.jpg', instance_np*50)
        return Image.fromarray(image_np), Image.fromarray(instance_np), Image.fromarray(label_np)

    def getCenterRadius(self, instance):
        instance_np = np.array(instance)
        ids = np.unique(instance_np)[1:]
        centers, radius, uids = [], [], []
        vMax, uMax = self.pad_size
        for uid in ids:
            inst_mask = instance_np == uid
            vs_, us_ = np.nonzero(inst_mask)
            vc, uc = vs_.mean(), us_.mean()
            vr, ur = np.abs(vs_-vc).max(), np.abs(us_-uc).max()
            centers.append([vc/vMax, uc/uMax])
            radius.append([vr/vMax + 1e-5, ur/uMax + 1e-5])
            uids.append(uid)
        return centers, radius, uids

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

            # Load the reference sample
            index2 = random.randint(0, self.real_size - 1)
            image2 = Image.open(self.mots_image_list[index2])
            instance2 = Image.open(self.mots_instance_list[index2])
            instance2, label2 = self.decode_instance(instance2)
            index3 = random.randint(0, self.real_size - 1)
            image3 = Image.open(self.mots_image_list[index3])
            instance3 = Image.open(self.mots_instance_list[index3])
            instance3, label3 = self.decode_instance(instance3)

            if random.random() > self.ori_prob:
                image, instance, label = self.randScaleAndFlip(image, instance, label)
                image2, instance2, label2 = self.randScaleAndFlip(image2, instance2, label2)
                image3, instance3, label3 = self.randScaleAndFlip(image3, instance3, label3)
                image, instance, label = self.copy_and_paste(image, instance, label, image2, instance2, label2)
                image, instance, label = self.copy_and_paste(image, instance, label, image3, instance3, label3)
            image, instance, label, pad_mask = self.pad_array([image, instance, label], self.pad_size)
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
        if self.centerRadius:
            centers, radius, ids = self.getCenterRadius(instance)
            sample['center'] = np.array(centers, dtype=np.float32)
            sample['radius'] = np.array(radius, dtype=np.float32)
            sample['ids'] = np.array(ids, dtype=np.float32)
        return sample

    def get_data_from_crops(self, index):
        index = random.randint(0, len(self.crop_image_list) - 1)
        sample = {}
        image = Image.open(self.crop_image_list[index])
        sample['im_name'] = self.crop_image_list[index]
        sample['im_shape'] = np.array([image.size])
        # load instances
        instance = Image.open(self.crop_instance_list[index])
        instance, label = self.decode_instance(instance)

        # Load the reference sample
        index2 = random.randint(0, self.real_size - 1)
        image2 = Image.open(self.mots_image_list[index2])
        instance2 = Image.open(self.mots_instance_list[index2])
        instance2, label2 = self.decode_instance(instance2)

        image, instance, label = self.randScaleAndFlip(image, instance, label)
        image2, instance2, label2 = self.randScaleAndFlip(image2, instance2, label2)
        image, instance, label = self.copy_and_paste(image, instance, label, image2, instance2, label2)
        image, instance, label, pad_mask = self.pad_array([image, instance, label], self.pad_size)

        sample['image'] = image
        sample['instance'] = instance
        sample['label'] = label
        sample['pad_mask'] = pad_mask
        if self.centerRadius:
            centers, radius, ids = self.getCenterRadius(instance)
            sample['center'] = np.array(centers, dtype=np.float32)
            sample['radius'] = np.array(radius, dtype=np.float32)
            sample['ids'] = np.array(ids, dtype=np.float32)
        return sample

    def __getitem__(self, index):
        if not self.requireObject:
            if (not self.add_crop) or (self.type in ['test', 'val']):
                sample = self.get_data_from_kins(index)
            elif self.add_crop and self.crop_only:
                sample = self.get_data_from_crops(index)
            else:
                sample = self.get_data_from_kins(index) if random.random() > 0.5 else self.get_data_from_crops(index)
                # sample = self.get_data_from_crops(index)
        else:
            while 1:
                if (not self.add_crop) or (self.type in ['test', 'val']):
                    sample = self.get_data_from_kins(index)
                elif self.add_crop and self.crop_only:
                    sample = self.get_data_from_crops(index)
                else:
                    sample = self.get_data_from_kins(index) if random.random() > 0.5 else self.get_data_from_crops(index)
                if sample['ids'].shape[0] > 0:
                    break
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
        # if self.type=='crop':
        #     assert mask.sum() > 0
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids
            class_map[mask] = 1

        # assign vehicles but not car to -2, dontcare
        mask = np.logical_and(pic > 0, pic < 1000)
        mask_others = (pic == 10000) | mask
        if mask_others.sum() > 0:
            class_map[mask_others] = -2

        return Image.fromarray(instance_map), Image.fromarray(class_map)

