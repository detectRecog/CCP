"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math
import torch
import torch.nn as nn
import models.erfnet as erfnet
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class BranchedERFNetUp4(nn.Module):
    def __init__(self, num_classes, input_channel=3, dec_DCN=False, vX=False, sam=False, unet=False, fc=5):
        super().__init__()

        print('Creating branched erfnet with {} classes'.format(num_classes))
        self.encoder = erfnet.Encoder(sum(num_classes), input_channel=input_channel)

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.DecoderHalf(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        # upscale images by two
        input = F.interpolate(input=input, scale_factor=(2.0, 2.0), mode='bicubic', align_corners=False)
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        output = torch.cat([decoder.forward(output) for decoder in self.decoders], 1)
        return output



def LocationEmbedding(f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    position_mat = torch.cat((cx, cy, w, h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, -1)
    position_mat = position_mat.view(f_g.shape[0], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(f_g.shape[0], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding