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
        # self.bn = nn.GroupNorm(8, noutput, eps=1e-6)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
    # 定义一系列的高斯核
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt) # 归一化操作，保证特征经过blur后信息总量不变
        # 非grad操作的参数利用buffer存储
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            # 利用固定参数的conv2d+stride实现blurpool
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class DownsamplerBlockBlur (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = BlurPool(ninput, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        # self.bn = nn.GroupNorm(8, noutput, eps=1e-6)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class DownsamplerBlock7 (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (7, 7), stride=2, padding=3, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        # self.bn = nn.GroupNorm(8, noutput, eps=1e-6)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class DownsamplerBlock5 (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (5, 5), stride=2, padding=2, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        # self.bn = nn.GroupNorm(8, noutput, eps=1e-6)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated, gn_groups=16):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        # self.bn1 = nn.GroupNorm(gn_groups, chann, eps=1e-6)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        # self.bn2 = nn.GroupNorm(gn_groups, chann, eps=1e-6)

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


class non_bottleneck_1d_IN (nn.Module):
    def __init__(self, chann, dropprob, dilated, gn_groups=16):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.InstanceNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

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


class non_bottleneck_1d_gn (nn.Module):
    def __init__(self, chann, dropprob, dilated, gn_groups=16):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.bn1 = nn.GroupNorm(gn_groups, chann, eps=1e-6)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

        # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.bn2 = nn.GroupNorm(gn_groups, chann, eps=1e-6)

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


class Encoder34(nn.Module):
    def __init__(self, input_channel=3, fc=5, c1=16, c2=64, c3=128, ot=256, blurPool=False):
        super().__init__()
        downpool = DownsamplerBlock if not blurPool else DownsamplerBlockBlur
        self.initial_block = downpool(input_channel, c1)

        self.layers = nn.ModuleList()

        self.layers.append(downpool(c1, c2))
        for x in range(0, fc):  # 5 times
            self.layers.append(non_bottleneck_1d(c2, 0.03, 1))

        self.layers.append(downpool(c2, c3))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 2))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 4))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 8))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 2))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 4))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 8))

        self.layers4 = nn.Sequential(
            downpool(c3, ot),
            non_bottleneck_1d(ot, 0.3, 2),
            non_bottleneck_1d(ot, 0.3, 4),
            non_bottleneck_1d(ot, 0.3, 8),
            non_bottleneck_1d(ot, 0.3, 2),
            non_bottleneck_1d(ot, 0.3, 4),
            non_bottleneck_1d(ot, 0.3, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
        output4 = self.layers4(output)
        return output, output4


class refine3(nn.Module):
    def __init__(self, ic=256, oc=256, dropout=0.3):
        super().__init__()
        self.refine_layer = nn.Sequential(
            nn.Conv2d(ic, oc, 1),
            non_bottleneck_1d(oc, dropout, 1),
            non_bottleneck_1d(oc, dropout, 2),
            non_bottleneck_1d(oc, dropout, 3),
            non_bottleneck_1d(oc, dropout, 1),
            non_bottleneck_1d(oc, dropout, 2),
            non_bottleneck_1d(oc, dropout, 3),
        )

    def forward(self, input):
        return self.refine_layer(input)


# class Decoder34 (nn.Module):
#     def __init__(self, num_classes, ic=256, c1=64, c2=32):
#         super().__init__()
#
#         self.lay1 = nn.Sequential(
#             non_bottleneck_1d_gn(ic, 0.0, 1),
#             nn.ConvTranspose2d(ic, c1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         )
#         self.lay2 = nn.Sequential(
#             nn.ConvTranspose2d(c1, c2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.GroupNorm(8, c2),
#             nn.LeakyReLU(0.1, inplace=True),
#             non_bottleneck_1d_gn(c2, 0.0, 1),
#         )
#         self.lay3 = nn.Sequential(
#             nn.ConvTranspose2d(c2, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.GroupNorm(8, 16),
#             nn.LeakyReLU(0.1, inplace=True),
#             non_bottleneck_1d_gn(16, 0.0, 1),
#         )
#
#         self.adapter1 = torch.nn.Conv2d(ic, c1, 1)
#         self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
#
#     def forward(self, input3, input4):
#         # input at 1/2^3 and 1/2^4
#         x = self.lay1(input4)
#
#         x = self.adapter1(input3) + x
#
#         x = self.lay2(x)
#         x = self.lay3(x)
#
#         return self.output_conv(x)


class Decoder34 (nn.Module):
    def __init__(self, num_classes, ic=256, c1=64, c2=32, ic2=256):
        super().__init__()

        self.lay1 = nn.Sequential(
            non_bottleneck_1d_gn(ic2, 0.0, 1),
            nn.ConvTranspose2d(ic2, c1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        )
        self.lay2 = nn.Sequential(
            non_bottleneck_1d_gn(2*c1, 0.0, 1),
            nn.ConvTranspose2d(2*c1, c2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, c2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.lay3 = nn.Sequential(
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.ConvTranspose2d(c2, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.adapter1 = torch.nn.Conv2d(ic, c1, 1)
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input3, input4):
        # input at 1/2^3 and 1/2^4
        x = self.lay1(input4)

        x = torch.cat([x, self.adapter1(input3)], dim=1)

        x = self.lay2(x)
        x = self.lay3(x)

        return self.output_conv(x)


class Decoder34ASFF (nn.Module):
    def __init__(self, num_classes, ic=256, c1=64, c2=32, ic2=512):
        super().__init__()

        self.compress_level_4 = nn.Sequential(
            nn.Conv2d(ic2, ic, 1, 1),
            nn.BatchNorm2d(ic)
        )
        compress_c = 16
        self.weight_level_4 = nn.Sequential(
            nn.Conv2d(ic, compress_c, 1, 1),
            nn.BatchNorm2d(compress_c)
        )
        self.weight_level_3 = nn.Sequential(
            nn.Conv2d(ic, compress_c, 1, 1),
            nn.BatchNorm2d(compress_c)
        )
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.expand = nn.Sequential(
            nn.Conv2d(ic, 2*c1, 1, 1),
            nn.BatchNorm2d(2*c1)
        )

        self.lay2 = nn.Sequential(
            non_bottleneck_1d_gn(2*c1, 0.0, 1),
            nn.ConvTranspose2d(2*c1, c2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, c2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.lay3 = nn.Sequential(
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.ConvTranspose2d(c2, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input3, input4):
        # input at 1/2^3 and 1/2^4
        input4_resized = F.interpolate(self.compress_level_4(input4), scale_factor=2, mode='nearest')
        input3_resized = input3

        level_3_weight_v = self.weight_level_3(input3_resized)
        level_4_weight_v = self.weight_level_4(input4_resized)
        levels_weight_v = torch.cat([level_3_weight_v, level_4_weight_v], 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input3_resized * levels_weight[:, 0:1, :, :] + input4_resized * levels_weight[:, 1:2, :, :]
        x = self.expand(fused_out_reduced)

        x = self.lay2(x)
        x = self.lay3(x)

        return self.output_conv(x)


class Decoder3 (nn.Module):
    def __init__(self, num_classes, ic=256, c2=32):
        super().__init__()
        self.lay2 = nn.Sequential(
            nn.Conv2d(ic, c2*2, 1),
            non_bottleneck_1d_gn(c2*2, 0.0, 1),
            nn.ConvTranspose2d(c2*2, c2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, c2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.lay3 = nn.Sequential(
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.ConvTranspose2d(c2, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input3):
        # input at 1/2^3 and 1/2^4
        x = self.lay2(input3)
        x = self.lay3(x)
        return self.output_conv(x)


class Encoder23(nn.Module):
    def __init__(self, input_channel=3, c1=16, c2=64, c3=128):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock7(input_channel, c1),
            non_bottleneck_1d(c1, 0.0, 1),
            non_bottleneck_1d(c1, 0.0, 2),
        )

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(c1, c2))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 1))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 2))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 4))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 8))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 1))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 2))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 4))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 8))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 1))


        self.layers4 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d(c3, 0.3, 1),
            non_bottleneck_1d(c3, 0.3, 2),
            non_bottleneck_1d(c3, 0.3, 4),
            non_bottleneck_1d(c3, 0.3, 8),
            non_bottleneck_1d(c3, 0.3, 1),
            non_bottleneck_1d(c3, 0.3, 2),
            non_bottleneck_1d(c3, 0.3, 4),
            non_bottleneck_1d(c3, 0.3, 8),
            non_bottleneck_1d(c3, 0.3, 1),
        )

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
        output3 = self.layers4(output)
        return output, output3


class Decoder23 (nn.Module):
    def __init__(self, num_classes, ic=256, c1=32, c2=16, ic2=256):
        super().__init__()

        self.lay1 = nn.Sequential(
            non_bottleneck_1d_gn(ic2, 0.0, 1),
            nn.ConvTranspose2d(ic2, c1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        )
        self.lay2 = nn.Sequential(
            non_bottleneck_1d_gn(2*c1, 0.0, 1),
            nn.ConvTranspose2d(2*c1, c2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, c2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.adapter1 = nn.Sequential(
            non_bottleneck_1d_gn(ic, 0.0, 1),
            torch.nn.Conv2d(ic, c1, 1)
        )
        self.output_conv = nn.ConvTranspose2d(c2, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input2, input3):
        # input at 1/2^3 and 1/2^4
        x = self.lay1(input3)

        x = torch.cat([x, self.adapter1(input2)], dim=1)

        x = self.lay2(x)

        return self.output_conv(x)



class Decoder2 (nn.Module):
    def __init__(self, num_classes, ic=256, c2=16):
        super().__init__()
        self.lay3 = nn.Sequential(
            non_bottleneck_1d(ic, 0.0, 1),
            nn.ConvTranspose2d(ic, c2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.output_conv = nn.ConvTranspose2d(c2, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        # input at 1/2^2
        x = self.lay3(x)
        return self.output_conv(x)


class Encoder234(nn.Module):
    def __init__(self, input_channel=3, c1=16, c2=64, c3=128, c4=256):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock7(input_channel, c1),
            non_bottleneck_1d(c1, 0.0, 1),
            non_bottleneck_1d(c1, 0.0, 2),
        )

        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(c1, c2))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 1))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 2))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 1))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 2))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 1))
        self.layers.append(non_bottleneck_1d(c2, 0.1, 2))


        self.layers3 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d(c3, 0.2, 1),
            non_bottleneck_1d(c3, 0.2, 2),
            non_bottleneck_1d(c3, 0.2, 4),
            non_bottleneck_1d(c3, 0.2, 1),
            non_bottleneck_1d(c3, 0.2, 2),
            non_bottleneck_1d(c3, 0.2, 4),
            non_bottleneck_1d(c3, 0.2, 1),
            non_bottleneck_1d(c3, 0.2, 2),
            non_bottleneck_1d(c3, 0.2, 4),
        )
        self.layers4 = nn.Sequential(
            DownsamplerBlock(c3, c4),
            non_bottleneck_1d(c4, 0.3, 1),
            non_bottleneck_1d(c4, 0.3, 2),
            non_bottleneck_1d(c4, 0.3, 4),
            non_bottleneck_1d(c4, 0.3, 1),
            non_bottleneck_1d(c4, 0.3, 2),
            non_bottleneck_1d(c4, 0.3, 4),
            non_bottleneck_1d(c4, 0.3, 1),
            non_bottleneck_1d(c4, 0.3, 2),
            non_bottleneck_1d(c4, 0.3, 4),
        )

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
        output3 = self.layers3(output)
        output4 = self.layers4(output3)
        return output, output3, output4


class Encoder345(nn.Module):
    def __init__(self, input_channel=3, fc=5, c1=16, c2=64, c3=128, c4=256, ot=256, d5=0.3):
        super().__init__()
        self.initial_block = DownsamplerBlock(input_channel, c1)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(c1, c2))
        for x in range(0, fc):  # 5 times
            self.layers.append(non_bottleneck_1d(c2, 0.03, 1))

        self.layers.append(DownsamplerBlock(c2, c3))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 2))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 4))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 8))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 2))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 4))
        self.layers.append(non_bottleneck_1d(c3, 0.2, 8))

        self.layers4 = nn.Sequential(
            DownsamplerBlock(c3, c4),
            non_bottleneck_1d(c4, 0.3, 2),
            non_bottleneck_1d(c4, 0.3, 4),
            non_bottleneck_1d(c4, 0.3, 8),
            non_bottleneck_1d(c4, 0.3, 12),
            non_bottleneck_1d(c4, 0.3, 2),
            non_bottleneck_1d(c4, 0.3, 4),
            non_bottleneck_1d(c4, 0.3, 8),
            non_bottleneck_1d(c4, 0.3, 12),
        )
        self.layers5 = nn.Sequential(
            DownsamplerBlock(c4, ot),
            non_bottleneck_1d(ot, d5, 2),
            non_bottleneck_1d(ot, d5, 4),
            non_bottleneck_1d(ot, d5, 6),
            non_bottleneck_1d(ot, d5, 8),
            non_bottleneck_1d(ot, d5, 2),
            non_bottleneck_1d(ot, d5, 4),
            non_bottleneck_1d(ot, d5, 6),
            non_bottleneck_1d(ot, d5, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
        output4 = self.layers4(output)
        output5 = self.layers5(output4)
        return output, output4, output5


class Encoder345InstanceNorm(nn.Module):
    def __init__(self, input_channel=3, fc=5, c1=16, c2=64, c3=128, c4=256, ot=256, nb = non_bottleneck_1d_IN):
        super().__init__()
        self.initial_block = DownsamplerBlock(input_channel, c1)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(c1, c2))
        for x in range(0, fc):  # 5 times
            self.layers.append(nb(c2, 0.03, 1))

        self.layers.append(DownsamplerBlock(c2, c3))
        self.layers.append(nb(c3, 0.2, 2))
        self.layers.append(nb(c3, 0.2, 4))
        self.layers.append(nb(c3, 0.2, 8))
        self.layers.append(nb(c3, 0.2, 2))
        self.layers.append(nb(c3, 0.2, 4))
        self.layers.append(nb(c3, 0.2, 8))

        self.layers4 = nn.Sequential(
            DownsamplerBlock(c3, c4),
            nb(c4, 0.3, 2),
            nb(c4, 0.3, 4),
            nb(c4, 0.3, 8),
            nb(c4, 0.3, 12),
            nb(c4, 0.3, 2),
            nb(c4, 0.3, 4),
            nb(c4, 0.3, 8),
            nb(c4, 0.3, 12),
        )
        self.layers5 = nn.Sequential(
            DownsamplerBlock(c4, ot),
            nb(ot, 0.3, 2),
            nb(ot, 0.3, 4),
            nb(ot, 0.3, 6),
            nb(ot, 0.3, 8),
            nb(ot, 0.3, 2),
            nb(ot, 0.3, 4),
            nb(ot, 0.3, 6),
            nb(ot, 0.3, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
        output4 = self.layers4(output)
        output5 = self.layers5(output4)
        return output, output4, output5


class Encoder0114(nn.Module):
    def __init__(self, input_channel=3, fc=5):
        super().__init__()
        self.initial_block = DownsamplerBlock(input_channel, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, fc):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        self.layers.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers.append(non_bottleneck_1d(128, 0.3, 12))
        self.layers.append(non_bottleneck_1d(128, 0.3, 16))
        self.layers.append(non_bottleneck_1d(128, 0.3, 1))
        self.layers.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers.append(non_bottleneck_1d(128, 0.3, 6))
        self.layers.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers.append(non_bottleneck_1d(128, 0.5, 1))
        self.layers.append(non_bottleneck_1d(128, 0.5, 2))
        self.layers.append(non_bottleneck_1d(128, 0.5, 3))
        self.layers.append(non_bottleneck_1d(128, 0.5, 4))
        self.layers.append(non_bottleneck_1d(128, 0.5, 1))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


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


class DecoderHalf (nn.Module):
    # twice height
    def __init__(self, num_classes, ic=128, oc=16):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(ic, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, oc))
        self.layers.append(non_bottleneck_1d(oc, 0, 1))
        self.layers.append(non_bottleneck_1d(oc, 0, 1))

        self.output_conv = nn.Conv2d(oc, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        outp = self.output_conv(output)

        return outp


class Encoder0117(nn.Module):
    def __init__(self, input_channel=3, fc=5):
        super().__init__()
        self.initial_block = DownsamplerBlock(input_channel, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, fc):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))
        self.layers.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers.append(non_bottleneck_1d(128, 0.3, 16))
        self.layers.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class Decoder0117 (nn.Module):
    # twice height
    def __init__(self, num_classes, ic=128, oc=16):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(ic, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, oc))
        self.layers.append(non_bottleneck_1d(oc, 0, 1))

        self.output_conv = nn.Conv2d(oc, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        outp = self.output_conv(output)

        return outp


class Encoder0118(nn.Module):
    def __init__(self, input_channel=3, c1=32, c2=128, c3=256, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 1)
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        return output2, output3


class Decoder0118 (nn.Module):
    def __init__(self, num_classes, ic=384, c2=32):
        super().__init__()
        self.lay3 = nn.Sequential(
            nn.Conv2d(ic, c2, 1),
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.GroupNorm(8, c2),
            nn.LeakyReLU(0.1, inplace=True),
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.ConvTranspose2d(c2, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        x = self.lay3(x)
        return self.output_conv(x)


class Decoder0118V2 (nn.Module):
    def __init__(self, num_classes, ic=64):
        super().__init__()
        self.lay3 = nn.Sequential(
            non_bottleneck_1d_gn(ic, 0.0, 1),
            nn.GroupNorm(8, ic),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(ic, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        )

    def forward(self, x):
        x = self.lay3(x)
        return x


class Encoder0119(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=192, c3=320, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        return output2, output3


class Encoder0120(nn.Module):
    def __init__(self, input_channel=3, c1=32, c2=128, c3=256, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 6),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 6),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.4, 1),
            non_bottleneck_1d(c2, 0.4, 2),
            non_bottleneck_1d(c2, 0.4, 4),
            non_bottleneck_1d(c2, 0.4, 6),
            non_bottleneck_1d(c2, 0.4, 8),
            non_bottleneck_1d(c2, 0.4, 1),
            non_bottleneck_1d(c2, 0.4, 2),
            non_bottleneck_1d(c2, 0.4, 4),
            non_bottleneck_1d(c2, 0.4, 6),
            non_bottleneck_1d(c2, 0.4, 8),
        )

        self.layers3 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        return output2, output3


class Encoder0121(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=256, c3=384, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

    def forward(self, input, fixHead=False):
        if fixHead:
            with torch.no_grad():
                output = self.initial_block(input)
        else:
            output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        return output2, output3


class Encoder0122(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=192, c3=384, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock7(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock5(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.4, 1),
            non_bottleneck_1d(c2, 0.4, 2),
            non_bottleneck_1d(c2, 0.4, 4),
            non_bottleneck_1d(c2, 0.4, 8),
            non_bottleneck_1d(c2, 0.4, 16),
            non_bottleneck_1d(c2, 0.4, 1),
            non_bottleneck_1d(c2, 0.4, 2),
            non_bottleneck_1d(c2, 0.4, 4),
            non_bottleneck_1d(c2, 0.4, 8),
            non_bottleneck_1d(c2, 0.4, 16),
        )

        self.layers3 = nn.Sequential(
            DownsamplerBlock5(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        return output2, output3


class Encoder0121GN(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=256, c3=384, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d_gn(c1, 0.03, 1),
            non_bottleneck_1d_gn(c1, 0.03, 2),
            non_bottleneck_1d_gn(c1, 0.03, 1),
            non_bottleneck_1d_gn(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d_gn(c2, 0.3, 1),
            non_bottleneck_1d_gn(c2, 0.3, 2),
            non_bottleneck_1d_gn(c2, 0.3, 4),
            non_bottleneck_1d_gn(c2, 0.3, 8),
            non_bottleneck_1d_gn(c2, 0.3, 16),
            non_bottleneck_1d_gn(c2, 0.3, 1),
            non_bottleneck_1d_gn(c2, 0.3, 2),
            non_bottleneck_1d_gn(c2, 0.3, 4),
            non_bottleneck_1d_gn(c2, 0.3, 8),
            non_bottleneck_1d_gn(c2, 0.3, 16),
            non_bottleneck_1d_gn(c2, 0.3, 1),
            non_bottleneck_1d_gn(c2, 0.3, 2),
            non_bottleneck_1d_gn(c2, 0.3, 4),
            non_bottleneck_1d_gn(c2, 0.3, 8),
            non_bottleneck_1d_gn(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d_gn(c3, d5, 1),
            non_bottleneck_1d_gn(c3, d5, 2),
            non_bottleneck_1d_gn(c3, d5, 4),
            non_bottleneck_1d_gn(c3, d5, 6),
            non_bottleneck_1d_gn(c3, d5, 8),
            non_bottleneck_1d_gn(c3, d5, 1),
            non_bottleneck_1d_gn(c3, d5, 2),
            non_bottleneck_1d_gn(c3, d5, 4),
            non_bottleneck_1d_gn(c3, d5, 6),
            non_bottleneck_1d_gn(c3, d5, 8),
        )

    def forward(self, input):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        return output2, output3


class ConvDownsampler (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput, (3, 3), stride=2, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.bn(self.conv(input))
        return F.relu(output)

class Encoder0126(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=256, c3=256, c4=64, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            ConvDownsampler(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 8),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 8),
        )

    def forward(self, input, low_level=False):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        if low_level:
            return output, output2, output3, output4
        return output2, output3, output4


class Encoder0202(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=256, c3=256, c4=64, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            ConvDownsampler(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 8),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 8),
        )

    def forward(self, input, low_level=False):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        if low_level:
            return output, output2, output3, output4
        return output2, output3, output4


class Decoder0202 (nn.Module):
    # deeper than 0118
    def __init__(self, num_classes, ic=384, c2=32):
        super().__init__()
        self.lay3 = nn.Sequential(
            nn.Conv2d(ic, c2, 1),
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.GroupNorm(8, c2),
            nn.LeakyReLU(0.1, inplace=True),
            non_bottleneck_1d_gn(c2, 0.0, 2),
            nn.GroupNorm(8, c2),
            nn.LeakyReLU(0.1, inplace=True),
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.ConvTranspose2d(c2, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        x = self.lay3(x)
        return self.output_conv(x)


class Encoder0203(nn.Module):
    def __init__(self, input_channel=3, c1=48, c2=224, c3=224, c4=48, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            ConvDownsampler(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 8),
        )

        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
        )

    def forward(self, input, low_level=False):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        if low_level:
            return output, output2, output3, output4
        return output2, output3, output4


class Decoder0203 (nn.Module):
    def __init__(self, num_classes, ic=384, c2=32):
        super().__init__()
        self.lay3 = nn.Sequential(
            nn.Conv2d(ic, c2, 1),
            non_bottleneck_1d_gn(c2, 0.0, 1),
            nn.ConvTranspose2d(c2, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        x = self.lay3(x)
        return self.output_conv(x)


class Encoder0213(nn.Module):
    # for MOT17, support whole frame input (1080+8, 1920)
    def __init__(self, input_channel=3, c1=64, c2=256, c3=256, c4=64, c5=64, d5=0.5, d4=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            ConvDownsampler(c2, c3),
            non_bottleneck_1d(c3, d4, 1),
            non_bottleneck_1d(c3, d4, 2),
            non_bottleneck_1d(c3, d4, 4),
            non_bottleneck_1d(c3, d4, 8),
            non_bottleneck_1d(c3, d4, 1),
            non_bottleneck_1d(c3, d4, 2),
            non_bottleneck_1d(c3, d4, 4),
            non_bottleneck_1d(c3, d4, 8),
        )

        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
        )

        self.layers5 = nn.Sequential(
            ConvDownsampler(c4, c5),
            non_bottleneck_1d(c5, d5, 1),
            non_bottleneck_1d(c5, d5, 2),
            non_bottleneck_1d(c5, d5, 4),
            non_bottleneck_1d(c5, d5, 6),
            non_bottleneck_1d(c5, d5, 1),
            non_bottleneck_1d(c5, d5, 2),
            non_bottleneck_1d(c5, d5, 4),
            non_bottleneck_1d(c5, d5, 6),
        )

    def forward(self, input, low_level=False):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        output5 = self.layers5(output4)
        if low_level:
            return output, output2, output3, output4, output5
        return output2, output3, output4, output5


class Encoder0218_1500(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=256, c3=256, c4=64, c5=64, d4=0.4, d5=0.5, d3=0.3):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.1, 1),
            non_bottleneck_1d(c2, 0.1, 2),
            non_bottleneck_1d(c2, 0.1, 4),
            non_bottleneck_1d(c2, 0.1, 8),
        )

        self.layers3 = nn.Sequential(
            ConvDownsampler(c2, c3),
            non_bottleneck_1d(c3, d3, 1),
            non_bottleneck_1d(c3, d3, 2),
            non_bottleneck_1d(c3, d3, 4),
            non_bottleneck_1d(c3, d3, 8),
            non_bottleneck_1d(c3, d3, 16),
            non_bottleneck_1d(c3, d3, 1),
            non_bottleneck_1d(c3, d3, 2),
            non_bottleneck_1d(c3, d3, 4),
            non_bottleneck_1d(c3, d3, 8),
            non_bottleneck_1d(c3, d3, 16),
            non_bottleneck_1d(c3, d3, 1),
            non_bottleneck_1d(c3, d3, 2),
            non_bottleneck_1d(c3, d3, 4),
            non_bottleneck_1d(c3, d3, 8),
            non_bottleneck_1d(c3, d3, 16),
        )

        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, d4, 1),
            non_bottleneck_1d(c4, d4, 2),
            non_bottleneck_1d(c4, d4, 4),
            non_bottleneck_1d(c4, d4, 8),
            non_bottleneck_1d(c4, d4, 16),
            non_bottleneck_1d(c4, d4, 1),
            non_bottleneck_1d(c4, d4, 2),
            non_bottleneck_1d(c4, d4, 4),
            non_bottleneck_1d(c4, d4, 8),
            non_bottleneck_1d(c4, d4, 16),
        )

        self.layers5 = nn.Sequential(
            ConvDownsampler(c4, c5),
            non_bottleneck_1d(c5, d5, 1),
            non_bottleneck_1d(c5, d5, 2),
            non_bottleneck_1d(c5, d5, 4),
            non_bottleneck_1d(c5, d5, 8),
            non_bottleneck_1d(c5, d5, 16),
            non_bottleneck_1d(c5, d5, 1),
            non_bottleneck_1d(c5, d5, 2),
            non_bottleneck_1d(c5, d5, 4),
            non_bottleneck_1d(c5, d5, 8),
            non_bottleneck_1d(c5, d5, 16),
        )

    def forward(self, input, low_level=False):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        output5 = self.layers5(output4)
        if low_level:
            return output, output2, output3, output4, output5
        return output2, output3, output4, output5


class Encoder0218_1600(nn.Module):
    def __init__(self, input_channel=3, c1=32, c2=128, c3=256, c4=128, c5=64, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
        )
        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.03, 1),
            non_bottleneck_1d(c2, 0.03, 2),
        )
        self.layers3 = nn.Sequential(
            DownsamplerBlock(c2, c3),
            non_bottleneck_1d(c3, 0.3, 1),
            non_bottleneck_1d(c3, 0.3, 2),
            non_bottleneck_1d(c3, 0.3, 4),
            non_bottleneck_1d(c3, 0.3, 8),
            non_bottleneck_1d(c3, 0.3, 16),
            non_bottleneck_1d(c3, 0.3, 1),
            non_bottleneck_1d(c3, 0.3, 2),
            non_bottleneck_1d(c3, 0.3, 4),
            non_bottleneck_1d(c3, 0.3, 8),
            non_bottleneck_1d(c3, 0.3, 16),
            non_bottleneck_1d(c3, 0.3, 1),
            non_bottleneck_1d(c3, 0.3, 2),
            non_bottleneck_1d(c3, 0.3, 4),
            non_bottleneck_1d(c3, 0.3, 8),
            non_bottleneck_1d(c3, 0.3, 16),
        )
        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, 0.4, 1),
            non_bottleneck_1d(c4, 0.4, 2),
            non_bottleneck_1d(c4, 0.4, 4),
            non_bottleneck_1d(c4, 0.4, 8),
            non_bottleneck_1d(c4, 0.4, 16),
            non_bottleneck_1d(c4, 0.4, 1),
            non_bottleneck_1d(c4, 0.4, 2),
            non_bottleneck_1d(c4, 0.4, 4),
            non_bottleneck_1d(c4, 0.4, 8),
            non_bottleneck_1d(c4, 0.4, 16),
        )

        self.layers5 = nn.Sequential(
            ConvDownsampler(c4, c5),
            non_bottleneck_1d(c5, d5, 1),
            non_bottleneck_1d(c5, d5, 2),
            non_bottleneck_1d(c5, d5, 4),
            non_bottleneck_1d(c5, d5, 8),
            non_bottleneck_1d(c5, d5, 16),
            non_bottleneck_1d(c5, d5, 1),
            non_bottleneck_1d(c5, d5, 2),
            non_bottleneck_1d(c5, d5, 4),
            non_bottleneck_1d(c5, d5, 8),
            non_bottleneck_1d(c5, d5, 16),
        )

    def forward(self, input, low_level=False):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        output5 = self.layers5(output4)
        return output2, output3, output4, output5


class Encoder0126_dilate(nn.Module):
    def __init__(self, input_channel=3, c1=64, c2=256, c3=256, c4=64, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d_IN(c1, 0.03, 1),
            non_bottleneck_1d_IN(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            ConvDownsampler(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 16),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 16),
        )

        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 8),
            non_bottleneck_1d(c4, d5, 16),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 8),
            non_bottleneck_1d(c4, d5, 16),
        )

    def forward(self, input, low_level=False):
        output = self.initial_block(input)
        output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        if low_level:
            return output, output2, output3, output4
        return output2, output3, output4


class Encoder0126CCP(nn.Module):
    # fix some layers to save Memory
    def __init__(self, input_channel=3, c1=64, c2=256, c3=256, c4=64, d5=0.5):
        super().__init__()
        self.initial_block = nn.Sequential(
            DownsamplerBlock(input_channel, c1),
            non_bottleneck_1d(c1, 0.03, 1),
            non_bottleneck_1d(c1, 0.03, 2),
        )

        self.layers2 = nn.Sequential(
            DownsamplerBlock(c1, c2),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
            non_bottleneck_1d(c2, 0.3, 1),
            non_bottleneck_1d(c2, 0.3, 2),
            non_bottleneck_1d(c2, 0.3, 4),
            non_bottleneck_1d(c2, 0.3, 8),
            non_bottleneck_1d(c2, 0.3, 16),
        )

        self.layers3 = nn.Sequential(
            ConvDownsampler(c2, c3),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
            non_bottleneck_1d(c3, d5, 1),
            non_bottleneck_1d(c3, d5, 2),
            non_bottleneck_1d(c3, d5, 4),
            non_bottleneck_1d(c3, d5, 6),
            non_bottleneck_1d(c3, d5, 8),
        )

        self.layers4 = nn.Sequential(
            ConvDownsampler(c3, c4),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 8),
            non_bottleneck_1d(c4, d5, 1),
            non_bottleneck_1d(c4, d5, 2),
            non_bottleneck_1d(c4, d5, 4),
            non_bottleneck_1d(c4, d5, 6),
            non_bottleneck_1d(c4, d5, 8),
        )

    def forward(self, input, low_level=False):
        with torch.no_grad():
            output = self.initial_block(input)
            output2 = self.layers2(output)
        output3 = self.layers3(output2)
        output4 = self.layers4(output3)
        if low_level:
            return output, output2, output3, output4
        return output2, output3, output4