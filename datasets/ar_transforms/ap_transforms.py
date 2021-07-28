import numpy as np
import torch
from torchvision import transforms as tf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageStat, ImageFilter
import random


class AdjustBrightnessAdaptive(object):

    def __init__(self, level=1, prob=0.5):
        self.level = level
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            return image

        temp = image.convert('L')
        stat = ImageStat.Stat(temp)
        brightness = (stat.mean[0] / 255)

        # Think this makes more sense
        enhancer = ImageEnhance.Brightness(image)
        if 0.25 < brightness < 0.75:
            t = random.random()
            if t < 0.33:
                image = enhancer.enhance(0.1 * random.randint(15, 20))
            else: # large prob to decrease lightness
                image = enhancer.enhance(0.1 * random.randint(4, 8))

        return image


def get_default_ap_transforms(cj=True, cj_bri=0.5, cj_con=0.0, cj_hue=0.0, cj_sat=0.0, gamma=False, gblur=True):
    transforms = [ToPILImage()]
    if cj:
        transforms.append(ColorJitter(brightness=cj_bri,
                                      contrast=cj_con,
                                      saturation=cj_sat,
                                      hue=cj_hue))
    if gblur:
        transforms.append(RandomGaussianBlur(0.5, 5))
    transforms.append(ToTensor())
    if gamma:
        transforms.append(RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True))
    return tf.Compose(transforms)

def get_blur_gamma_color_transform(max_blur=7, blur_prob=0.5, cj_con=(0.4, 1.3)):
    transforms = [ToPILImage(),
                  ColorJitter(contrast=cj_con),
                  AdjustBrightnessAdaptive(),
                  RandomGaussianBlur(blur_prob, max_blur)]
    return tf.Compose(transforms)

def get_blur_gamma_transform(max_blur=7, blur_prob=0.5, min_gamma=0.7, max_gamma=1.5):
    transforms = [RandomGaussianBlur(blur_prob, max_blur), RandomGammaPIL(min_gamma=min_gamma, max_gamma=max_gamma, clip_image=True)]
    return tf.Compose(transforms)

def get_ap_transforms(cfg):
    transforms = [ToPILImage()]
    if cfg.cj:
        transforms.append(ColorJitter(brightness=cfg.cj_bri,
                                      contrast=cfg.cj_con,
                                      saturation=cfg.cj_sat,
                                      hue=cfg.cj_hue))
    if cfg.gblur:
        transforms.append(RandomGaussianBlur(0.5, 3))
    transforms.append(ToTensor())
    if cfg.gamma:
        transforms.append(RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True))
    return tf.Compose(transforms)


# from https://github.com/visinf/irr/blob/master/datasets/transforms.py
class ToPILImage(tf.ToPILImage):
    def __call__(self, imgs):
        # return [super(ToPILImage, self).__call__(im) for im in imgs]
        return super(ToPILImage, self).__call__(imgs)


class ColorJitter(tf.ColorJitter):
    def __call__(self, imgs):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        # return [transform(im) for im in imgs]
        return transform(imgs)


class ToTensor(tf.ToTensor):
    def __call__(self, imgs):
        # return [super(ToTensor, self).__call__(im) for im in imgs]
        return super(ToTensor, self).__call__(imgs)


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    # def __call__(self, imgs):
    #     gamma = self.get_params(self._min_gamma, self._max_gamma)
    #     return [self.adjust_gamma(im, gamma, self._clip_image) for im in imgs]
    def __call__(self, imgs):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(imgs, gamma, self._clip_image)

class RandomGammaPIL():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        image_np = np.array(image) / 255.0
        adjusted = np.power(image_np, gamma)
        if clip_image:
            adjusted = np.clip(adjusted, 0.0, 1.0)
        adjusted *= 255.0
        return Image.fromarray(adjusted.astype(np.uint8))

    def __call__(self, imgs):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(imgs, gamma, self._clip_image)


class RandomGaussianBlur():
    def __init__(self, p, max_k_sz):
        self.p = p
        self.max_k_sz = max_k_sz

    def __call__(self, imgs):
        if np.random.random() < self.p:
            radius = np.random.uniform(0, self.max_k_sz)
            # imgs = [im.filter(ImageFilter.GaussianBlur(radius)) for im in imgs]
            imgs = imgs.filter(ImageFilter.GaussianBlur(radius))
        return imgs
