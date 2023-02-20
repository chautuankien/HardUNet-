import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from hardnet_68 import hardnet

__all__ = ['NestedUNet']

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        #######################################################################
        kernel_size = 7
        #######################################################################

        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.spatial = spatial
        if not spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.cbam = CBAM(middle_channels, 8)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)



    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)



        return out
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        mid_ch = int(out_channels/2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.Conv2d(mid_ch, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU()
        #self.cbam = CBAM(in_channels, 8)

    def forward(self, x):
        #incbam = self.cbam(x)
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)

        return out

class VGG_dilated(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dilation):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels, middle_channels, 3, padding= 1)
        self.bn0 = nn.BatchNorm2d(middle_channels)

        self.conv1 = nn.Conv2d(middle_channels, middle_channels, 3, padding= dilation, dilation= dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #self.cbam1 = CBAM(in_channels, 8)
        #self.cbam2 = CBAM(out_channels, 8)

    def forward(self, x):
        #out = self.cbam1(x)
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        #out = self.relu(out)

        #out = self.cbam2(out)
        return out

'''
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 320, 640, 1024]
        dilation = [1, 2, 3, 4]
        self.deep_supervision = deep_supervision
        dilate_ch = int(nb_filter[4]/4)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2 = ResBlock(nb_filter[0]*3, nb_filter[0], nb_filter[0])
        self.conv0_3 = ResBlock(nb_filter[0]*4, nb_filter[0], nb_filter[0])
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv4_0 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[0])
        self.conv4_1 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[1])
        self.conv4_2 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[2])
        self.conv4_3 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[3])

        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[0], nb_filter[0])
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[1], nb_filter[1])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[2], nb_filter[3])

        self.conv1_2 = ResBlock(nb_filter[0]+nb_filter[1]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[1]+nb_filter[3], nb_filter[1], nb_filter[2])

        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[0]*2+nb_filter[2], nb_filter[0], nb_filter[1])


        self.hardnet = hardnet(arch=68)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        hardnetout = self.hardnet(input)

        x0_0 = hardnetout[0]

        x1_0 = hardnetout[1]

        x2_0 = hardnetout[2]

        x3_0 = hardnetout[3]

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x4_0 = self.conv4_0(x3_0)
        x4_1 = self.conv4_1(x3_0)
        x4_2 = self.conv4_2(x3_0)
        x4_3 = self.conv4_3(x3_0)


        x4 = torch.cat([x4_0, x4_1, x4_2, x4_3], 1)

        x3_1 = self.conv3_1(torch.cat([x3_0, x4], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)],1))
        x0_2 = self.conv0_2(torch.cat([x0_0,x0_1, self.up(x1_1)],1))
        x0_3 = self.conv0_3(torch.cat([x0_0,x0_1,x0_2,self.up(x1_2)],1))
        x0_4 = self.conv0_4(torch.cat([x0_0,x0_1,x0_2,x0_3, self.up(x1_3)],1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(self.up(x0_4))
            return output
'''

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        nb_filter = [64, 128, 320, 640, 1024]
        dilation = [1, 2, 3, 5]
        dilate_ch = int(nb_filter[4]/4)
        self.deep_supervision = deep_supervision
        cbam_ch = [128, 320, 640, 1024]
        self.cbam1 = CBAM(cbam_ch[0],8)
        self.cbam2 = CBAM(cbam_ch[1],8)
        self.cbam3 = CBAM(cbam_ch[2],16)
        self.cbam4 = CBAM(cbam_ch[3],32)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv4_0 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[0])
        self.conv4_1 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[1])
        self.conv4_2 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[2])
        self.conv4_3 = VGG_dilated(nb_filter[3], dilate_ch, dilate_ch, dilation[3])


        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = ResBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.hardnet = hardnet(arch=68)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        hardnetout = self.hardnet(input)

        x1_0 = hardnetout[1]
        #x1_0 = self.cbam1(x1_0)

        x2_0 = hardnetout[2]
        #x2_0 = self.cbam2(x2_0)

        x3_0 = hardnetout[3]
        #x3_0 = self.cbam3(x3_0)

        #x4 = hardnetout[4]


        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x4_0 = self.conv4_0(x3_0)
        x4_1 = self.conv4_1(x3_0)
        x4_2 = self.conv4_2(x3_0)
        x4_3 = self.conv4_3(x3_0)


        x4 = torch.cat([x4_0, x4_1, x4_2, x4_3], 1)
        #x4 = self.cbam4(x4)

        x3_1 = self.conv3_1(torch.cat([x3_0, x4], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(self.up(x1_3))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(self.up(x0_4))
            return output
