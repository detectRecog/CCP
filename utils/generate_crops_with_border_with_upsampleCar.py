"""
only for car, crop should be upsample for data distribution
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm
from config import kittiRoot
import multiprocessing
import pickle
from random import random
from file_utils import remove_and_mkdir


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def mkdir_if_no(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def is_image_file(filename, suffix=None):
    if suffix is not None:
        IMG_EXTENSIONS = [suffix]
    else:
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npz'
        ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, suffix=None, max=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, suffix):
                path = os.path.join(root, fname)
                images.append(path)
    if max is not None:
        return images[:max]
    else:
        return images


def low_density(arr):
    occupy = (arr > 1).sum()
    total = (arr >=0).sum()
    return occupy/total < max_occupy


def has_bg(arr):
    arr = np.array(arr)
    return (arr==0).sum() > 0


def process(tup):
    image_path, instance_path, name = tup['img'], tup['inst'], tup['name']
    kins_image = name[:4]=='KINS'
    if class_id == 2:
        OBJ_ID = class_id
    else:
        # for cars, use the line below
        OBJ_ID = 1
    # OBJ_ID = class_id

    image = Image.open(image_path)
    instance = Image.open(instance_path)
    w, h = image.size
    density = np.zeros((h,w)).astype(np.long)

    instance_np = np.array(instance, copy=False)
    object_mask = np.logical_and(instance_np >= OBJ_ID * 1000, instance_np < (OBJ_ID + 1) * 1000)

    ids = np.unique(instance_np[object_mask])
    if kins_image and ids.shape[0]<1:
        # dump unrelated
        return
    ids = ids[ids!= 0]
    np.random.shuffle(ids)

    # loop over instances, get center and border
    for j, id in enumerate(ids):

        y, x = np.where(instance_np == id)
        ymin, ymax = y.min(), y.max()
        xmin, xmax = x.min(), x.max()
        insth = ymax-ymin+1
        ym, xm = np.mean(y), np.mean(x)

        CROP_SIZE = CROP_SIZE_
        tsize = None
        if kins_image:
            if insth <= 5:
                continue
            elif insth < 30:
                # h<30 KINS instances should be upsampled by 70%, 1/1.7
                tsize = (CROP_SIZE_[1], CROP_SIZE_[0])
                CROP_SIZE = (np.array(CROP_SIZE_) * dratio).astype(np.int).tolist()
            else:
                pass

        ii = int(np.clip(ym-CROP_SIZE[0]/2, 0, h-CROP_SIZE[0]))
        jj = int(np.clip(xm-CROP_SIZE[1]/2, 0, w-CROP_SIZE[1]))

        im_crop = image.crop((jj, ii, jj + CROP_SIZE[1], ii + CROP_SIZE[0]))
        instance_crop = instance.crop((jj, ii, jj + CROP_SIZE[1], ii + CROP_SIZE[0]))
        if low_density(density[ii:ii + CROP_SIZE[0], jj:jj + CROP_SIZE[1]]):
            density[ii:ii + CROP_SIZE[0], jj:jj + CROP_SIZE[1]] += 1
            iname = name + '_' + str(j) + '.png'
            if tsize is not None:
                im_crop, instance_crop = im_crop.resize(tsize, resample=Image.BILINEAR), instance_crop.resize(tsize, resample=Image.NEAREST)
            im_crop.save(os.path.join(image_dir, iname))
            instance_crop.save(os.path.join(inst_dir, iname))

        # crop border if possible: left, right, upper
        # if 0:
        if random() < occlude_ratio:
            # check if left can crop
            xcrop = xmin + max(int((xmax-xmin)/occ_percent), 8)
            if xcrop > CROP_SIZE[1]:
                im_crop = image.crop((xcrop-CROP_SIZE[1], ii, xcrop, ii + CROP_SIZE[0]))
                instance_crop = instance.crop((xcrop-CROP_SIZE[1], ii, xcrop, ii + CROP_SIZE[0]))
                if low_density(density[ii:ii + CROP_SIZE[0], xcrop - CROP_SIZE[1]:xcrop]):
                    density[ii:ii+CROP_SIZE[0],xcrop-CROP_SIZE[1]:xcrop] += 1
                    iname = name + '_' + str(j) + '_left.png'
                    if tsize is not None:
                        im_crop, instance_crop = im_crop.resize(tsize, resample=Image.BILINEAR), instance_crop.resize(tsize, resample=Image.NEAREST)
                    im_crop.save(os.path.join(image_dir, iname))
                    instance_crop.save(os.path.join(inst_dir, iname))
        # if 0:
        if random() < occlude_ratio:
            # check if right can crop
            xcrop = xmax - max(int((xmax - xmin) / occ_percent), 8)
            if w - xcrop > CROP_SIZE[1]:
                im_crop = image.crop((xcrop, ii, xcrop + CROP_SIZE[1], ii + CROP_SIZE[0]))
                instance_crop = instance.crop((xcrop, ii, xcrop + CROP_SIZE[1], ii + CROP_SIZE[0]))
                if low_density(density[ii:ii + CROP_SIZE[0], xcrop:xcrop + CROP_SIZE[1]]):
                    density[ii:ii + CROP_SIZE[0], xcrop:xcrop + CROP_SIZE[1]] += 1
                    iname = name + '_' + str(j) + '_right.png'
                    if tsize is not None:
                        im_crop, instance_crop = im_crop.resize(tsize, resample=Image.BILINEAR), instance_crop.resize(tsize, resample=Image.NEAREST)
                    im_crop.save(os.path.join(image_dir, iname))
                    instance_crop.save(os.path.join(inst_dir, iname))

    if not kins_image:
        CROP_SIZE = CROP_SIZE_
        # for image border, left-down and right-down
        im_crop = image.crop((0, (h-CROP_SIZE[0])//2, CROP_SIZE[1], (h+CROP_SIZE[0])//2))
        instance_crop = instance.crop((0, (h-CROP_SIZE[0])//2, CROP_SIZE[1], (h+CROP_SIZE[0])//2))
        if low_density(density[(h-CROP_SIZE[0])//2:(h+CROP_SIZE[0])//2, 0:CROP_SIZE[1]]) and has_bg(instance_crop):
            density[(h-CROP_SIZE[0])//2:(h+CROP_SIZE[0])//2, 0:CROP_SIZE[1]] += 1
            iname = name + '_' + 'left_down.png'
            im_crop.save(os.path.join(image_dir, iname))
            instance_crop.save(os.path.join(inst_dir, iname))

        im_crop = image.crop((w-CROP_SIZE[1], (h-CROP_SIZE[0])//2, w, (h+CROP_SIZE[0])//2))
        instance_crop = instance.crop((w-CROP_SIZE[1], (h-CROP_SIZE[0])//2, w, (h+CROP_SIZE[0])//2))
        if low_density(density[(h-CROP_SIZE[0])//2:(h+CROP_SIZE[0])//2, w-CROP_SIZE[1]:w])  and has_bg(instance_crop):
            density[(h-CROP_SIZE[0])//2:(h+CROP_SIZE[0])//2, w-CROP_SIZE[1]:w] += 1
            iname = name + '_' + 'right_down.png'
            im_crop.save(os.path.join(image_dir, iname))
            instance_crop.save(os.path.join(inst_dir, iname))


if __name__ == '__main__':
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    # dst_kins = 'KINS'
    # class_id = 26
    # CROP_SIZE_ = (192, 640)  # h, w for car
    # occlude_ratio = 0.33
    # occ_percent = 8
    # max_occupy = 0.8
    # dratio = 1/1.6

    # 0904 !!! awake from ideas, make some improvements now!
    dst_kins = 'KINS'
    class_id = 26
    # (224, 800) is suitable for training on 2 samples on one 11G card
    CROP_SIZE_ = (224, 800)  # h, w for car
    occlude_ratio = 0.7
    occ_percent = 8
    max_occupy = 0.75
    dratio = 1/1.6

    # initialize folders to save crops
    save_root = os.path.join(kittiRoot,'crop_' + dst_kins)
    remove_and_mkdir(save_root)
    inst_dir = os.path.join(save_root, 'instances')
    mkdir_if_no(inst_dir)
    image_dir = os.path.join(save_root, 'images')
    mkdir_if_no(image_dir)

    # load images from KINS
    instance_list = make_dataset(os.path.join(kittiRoot, 'training/'+dst_kins), suffix='.png') + make_dataset(os.path.join(kittiRoot, 'testing/'+dst_kins), suffix='.png')
    image_list = [el.replace(dst_kins, 'image_2') for el in instance_list]
    save_list = ['KINS_' + el.split('/')[-3] + '_' + el.split('/')[-1][:-4] for el in instance_list]

    # load images from KITTI MOTS
    mots_instance_root = os.path.join(kittiRoot, 'instances')
    mots_image_root = os.path.join(kittiRoot, 'images')
    mots_persons = load_pickle(os.path.join(kittiRoot, 'mots_inst_train5.pkl'))
    # mots_persons = load_pickle(os.path.join(kittiRoot, 'mots_inst_train.pkl'))

    mots_instance_list = [os.path.join(mots_instance_root, el) for el in mots_persons]
    mots_image_list =  [el.replace('instances', 'images') for el in mots_instance_list]
    mots_save_list = ['MOTS_' + '_'.join(el[:-4].split('/')[-2:]) for el in mots_instance_list]

    total_image_list = image_list + mots_image_list
    total_inst_list = instance_list + mots_instance_list
    total_save_list = save_list + mots_save_list
    infos = [{'img': total_image_list[el], 'inst': total_inst_list[el], 'name': total_save_list[el]} for el in range(len(total_image_list))]

    # for el in infos:
    #     process(el)
    pool = multiprocessing.Pool(processes=32)
    results = pool.map(process, infos)
    pool.close()
