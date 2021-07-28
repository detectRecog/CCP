# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class DownsamplerBlockDCN (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        from mmcv.ops import ModulatedDeformConv2dPack
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.dconv = ModulatedDeformConv2dPack(noutput, noutput, (3, 3), stride=1, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.dconv(output)
        output = self.bn(output)
        return F.relu(output)


class UpsamplerBlockDCN (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        from mmcv.ops import ModulatedDeformConv2dPack
        self.conv0 = ModulatedDeformConv2dPack(ninput, ninput, (3, 3), stride=1, padding=1, bias=True)
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(self.conv0(input))
        output = self.bn(output)
        return F.relu(output)


class DownsamplerBlock5(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (5, 5), stride=2, padding=2, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d_inst (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.InstanceNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class DecoderHalf (nn.Module):
    # twice height
    def __init__(self, num_classes, ic=128):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(ic, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.Conv2d(16, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        outp = self.output_conv(output)

        return outp


class Encoder(nn.Module):
    def __init__(self, num_classes, input_channel=3, fc=5, sc=2):
        super().__init__()
        self.initial_block = DownsamplerBlock(input_channel, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, fc):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, sc):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False, feat=False):
        output = self.initial_block(input)
        feats = output

        for layer in self.layers:
            output = layer(output)

        if predict:
            att = self.output_conv(output)
            return output, att

        if feat:
            return output, feats
        return output
