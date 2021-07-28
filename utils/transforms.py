"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageStat, ImageFilter
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import torch
import math
try:
    from pyblur import *
except:
    pass    # install pyblur following https://github.com/lospooky/pyblur/issues/5


class AdjustBrightness(object):

    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, sample):
        if random.random() < 0.66:
            return sample

        for k in sample.keys():
            if not 'image' in k:
                continue

            image = sample[k]
            temp = image.convert('L')
            stat = ImageStat.Stat(temp)
            brightness = (stat.mean[0] / 255)

            # Think this makes more sense
            enhancer = ImageEnhance.Brightness(image)
            if 0.25 < brightness < 0.75:
                t = random.random()
                if t < 0.5:
                    image = enhancer.enhance(0.1 * random.randint(15, 20))
                else:
                    image = enhancer.enhance(0.1 * random.randint(5, 8))

            sample[k] = image

        return sample


class AdjustBlurGamma(object):

    def __init__(self, keys=[], min_gamma=0.7, max_gamma=1.5, clip_image=True, max_k_sz=5, ori_prob=0.5):
        self.keys = keys
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image
        self.max_k_sz = max_k_sz
        self.ori_prob = ori_prob

    def get_params(self, min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    def __call__(self, sample):
        if random.random() < self.ori_prob:
            return sample

        for k in sample.keys():
            if not 'image' in k:
                continue

            image = sample[k]
            radius = np.random.uniform(0, self.max_k_sz)
            image = image.filter(ImageFilter.GaussianBlur(radius))

            image_np = np.array(image) / 255.0
            gamma = self.get_params(self._min_gamma, self._max_gamma)
            adjusted = np.power(image_np, gamma)
            if self._clip_image:
                adjusted = np.clip(adjusted, 0.0, 1.0)
            adjusted *= 255.0
            image = Image.fromarray(adjusted.astype(np.uint8))
            sample[k] = image

        return sample


def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring
    Args:
        kerneldim (int): size of the kernel used in motion blurring
    Returns:
        int: Random angle
    """
    kernelCenter = int(math.floor(kerneldim / 2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])


def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array
    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])


def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate
       blurring caused due to motion of camera.
    Args:
        img(NumPy Array): Input image with 3 channels
    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3, 5, 7, 9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes))
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:, :, i] = PIL2array1C(LinearMotionBlur(img[:, :, i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return np.array(blurred_img)


def AddGaussBlur(img, level=None):
    """
    添加高斯模糊, for (128, 64), 3 is very large and difficult
    """
    if level is None:
        level = random.randint(1, 3)
    choice = random.random()
    if choice < 0.7:
        return LinearMotionBlur3C(img)
    elif 0.7 <= choice < 0.85:
        return cv2.blur(img, (level * 2 + 1, level * 2 + 1))
    else:
        return cv2.medianBlur(img, level * 2 + 1)


class AddMotionBlur(object):

    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, sample):
        if random.random() < 0.66:
            return sample

        for k in sample.keys():
            if not 'image' in k:
                continue

            image = sample[k]
            # image.save('/data/xuzhenbo/1.jpg')
            imageBGR = np.array(image)[:,:,::-1]
            imageRGB = AddGaussBlur(imageBGR)[:,:,::-1]
            imageRGB = Image.fromarray(imageRGB, 'RGB')
            # imageRGB.save('/data/xuzhenbo/3.jpg')
            sample[k] = imageRGB

        return sample


class Flip(object):

    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, sample):
        if random.random() > 0.5:
            for idx, k in enumerate(self.keys):
                assert (k in sample)

                sample[k] = torch.flip(sample[k], [-1])
                if k == 'part0':
                    bg_mask = sample[k] == 0
                    sample[k] = 1.0 - sample[k]
                    sample[k][bg_mask] = 0
            sample['Flip'] = 1
        else:
            sample['Flip'] = 0
        return sample


class CropRandomObject:

    def __init__(self, keys=[],object_key="instance", size=100):
        self.keys = keys
        self.object_key = object_key
        self.size = size

    def __call__(self, sample):

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]
        
        if unique_objects.size > 0:
            random_id = np.random.choice(unique_objects, 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)
            
            i = int(np.clip(ym-self.size[1]/2, 0, h-self.size[1]))
            j = int(np.clip(xm-self.size[0]/2, 0, w-self.size[0]))

        else:
            i = random.randint(0, h - self.size[1])
            j = random.randint(0, w - self.size[0])

        for k in self.keys:
            assert(k in sample)

            sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        return sample


class LU_Pad(object):
    # pad at the right and the bottom
    def __init__(self, keys=[], size=100):
        self.keys = keys
        self.size = size

    def __call__(self, sample):

        for k in self.keys:

            assert(k in sample)

            w, h = sample[k].size

            padding = (0,0,self.size[1]-w, self.size[0]-h)

            sample[k] = F.pad(sample[k], padding) # change for MOTS20
            # sample[k] = F.pad(sample[k], padding, padding_mode='edge')

        sample['start_point'] = torch.FloatTensor([0,0])  # y0, x0
        sample['x_diff'] = torch.FloatTensor([0])
        return sample


class Reshape(object):
    # pad at the right and the bottom
    def __init__(self, keys=[], size=100):
        self.keys = keys
        self.size = size

    def __call__(self, sample):

        # rescale to self.size
        for k in self.keys:
            if 'image' in k:
                sample[k] = sample[k].resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
            elif 'instance' in k or 'label' in k:
                sample[k] = sample[k].resize((self.size[1], self.size[0]), resample=Image.NEAREST)
            else:
                raise NotImplementedError

        return sample


class RandomCrop(T.RandomCrop):

    def __init__(self, keys=[], size=100):

        super().__init__(size)
        self.keys = keys

    def __call__(self, sample):

        params = None

        for k in self.keys:

            assert(k in sample)

            if params is None:
                params = self.get_params(sample[k], self.size)

            sample[k] = F.crop(sample[k], *params)

        return sample


class RandomCropWithResize(T.RandomCrop):

    def __init__(self, keys=[], size=100):

        super().__init__(size)
        self.keys = keys

    def __call__(self, sample):

        params = None
        cropProb = random.random()

        if cropProb > 0.3:
            for k in self.keys:

                assert(k in sample)

                if params is None:
                    params = self.get_params(sample[k], self.size)

                sample[k] = F.crop(sample[k], *params)
        else:
            # rescale to self.size
            for k in self.keys:
                if 'image' in k:
                    sample[k] = sample[k].resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
                elif 'instance' in k or 'label' in k:
                    sample[k] = sample[k].resize((self.size[1], self.size[0]), resample=Image.NEAREST)
                else:
                    raise NotImplementedError
        return sample


class RandomCropWithResample(T.RandomCrop):

    def __init__(self, keys=[], size=100, min_ratio=0.7, max_ratio=1.1):
        # min_ratio=0.5, max_ratio=1.1 designed for person
        # y as large as possible
        super().__init__(size)
        self.keys = keys
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample):

        params = None
        cropProb = random.random()
        W,H = sample['image'].size[:2]
        if cropProb > 0.7:
            for k in self.keys:

                assert(k in sample)

                if params is None:
                    params = self.get_params(sample[k], self.size)

                sample[k] = F.crop(sample[k], *params)
        else:
            # crop and rescale to self.size
            resample_ratio = random.random() * (self.max_ratio-self.min_ratio) + self.min_ratio
            self.size_ = (int(self.size[0]*resample_ratio), int(self.size[1]*resample_ratio))
            for k in self.keys:
                if params is None:
                    y, x, h, w = self.get_params(sample[k], self.size_) # y,x,h,w
                    y = max(y, min(int(H*0.33), H-h-1))
                    params = (y, x, h, w)
                sample[k] = F.crop(sample[k], *params)
            for k in self.keys:
                if 'image' in k:
                    sample[k] = sample[k].resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
                elif 'instance' in k or 'label' in k:
                    sample[k] = sample[k].resize((self.size[1], self.size[0]), resample=Image.NEAREST)
                else:
                    raise NotImplementedError
        return sample


class RandomRotation(T.RandomRotation):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.resample, collections.Iterable):
            assert(len(keys) == len(self.resample))

    def __call__(self, sample):

        angle = self.get_params(self.degrees)

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            resample = self.resample
            if isinstance(resample, collections.Iterable):
                resample = resample[idx]

            sample[k] = F.rotate(sample[k], angle, resample,
                                 self.expand, self.center)

        return sample


class Resize(T.Resize):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.interpolation, collections.Iterable):
            assert(len(keys) == len(self.interpolation))

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, collections.Iterable):
                interpolation = interpolation[idx]

            sample[k] = F.resize(sample[k], self.size, interpolation)

        return sample


class ToTensor(object):

    def __init__(self, keys=[], type="float"):

        if isinstance(type, collections.Iterable):
            assert(len(keys) == len(type))

        self.keys = keys
        self.type = type

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            sample[k] = F.to_tensor(sample[k])

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]

            if t == torch.ByteTensor or t == torch.LongTensor:
                sample[k] = sample[k]*255

            sample[k] = sample[k].type(t)

        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']
        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
