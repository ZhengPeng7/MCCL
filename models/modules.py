import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from functools import partial
from einops import rearrange

from config import Config


config = Config()


class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, groups=0):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att == 'ASPP':
            self.dec_att = ASPP(channel_in=64)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(64)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        if config.dec_att:
            x = self.dec_att(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x


class _ASPPModule(nn.Module):
    def __init__(self, channel_in, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(channel_in, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, channel_in=64, output_stride=16):
        super(ASPP, self).__init__()
        self.down_scale = 1
        self.channel_inter = 256 // self.down_scale
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(channel_in, self.channel_inter, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(channel_in, self.channel_inter, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.channel_inter),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.channel_inter * 5, channel_in, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class GWM(nn.Module):
    def __init__(self, channel_in, num_groups1=8, num_groups2=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(GWM, self).__init__()
        self.num_heads = num_heads
        self.num_groups1 = num_groups1
        self.num_groups2 = num_groups2
        head_dim = channel_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(channel_in, channel_in * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channel_in, channel_in)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, recursive_index=False):
        B, N, C = x.shape
        if recursive_index == False:
            num_groups = self.num_groups1
        else:
            num_groups = self.num_groups2
            if num_groups != 1:
                idx = torch.randperm(N)
                x = x[:,idx,:]
                inverse = torch.argsort(idx)
        qkv = self.qkv(x).reshape(B, num_groups, N // num_groups, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)  
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_groups, N // num_groups, C)
        x = x.permute(0, 3, 1, 2).reshape(B, C, N).transpose(1, 2)
        if recursive_index == True and num_groups != 1:
            x = x[:,inverse,:]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CoAttLayer(nn.Module):
    def __init__(self, channel_in=512):
        super(CoAttLayer, self).__init__()

        self.all_attention = GCAM(channel_in)
    
    def forward(self, x):
        if self.training:
            if config.loadN > 1:
                channel_per_class = x.shape[0] // config.loadN
                x_per_class_corr_list = []
                for idx in range(0, x.shape[0], channel_per_class):
                    x_per_class = x[idx:idx+channel_per_class]

                    x_new_per_class = self.all_attention(x_per_class)

                    x_per_class_proto = torch.mean(x_new_per_class, (0, 2, 3), True).view(1, -1)
                    x_per_class_proto = x_per_class_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

                    x_per_class_corr = x_per_class * x_per_class_proto
                    x_per_class_corr_list.append(x_per_class_corr)
                weighted_x = torch.cat(x_per_class_corr_list, dim=0)
            else:
                x_new = self.all_attention(x)
                x_proto = torch.mean(x_new, (0, 2, 3), True).view(1, -1)
                x_proto = x_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1
                weighted_x = x * x_proto

        else:
            x_new = self.all_attention(x)
            x_proto = torch.mean(x_new, (0, 2, 3), True).view(1, -1)
            x_proto = x_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1
            weighted_x = x * x_proto
        return weighted_x


class GCAM(nn.Module):
    def __init__(self, channel_in=512):

        super(GCAM, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)


    def forward(self, x):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x.size()

        x_query = self.query_transform(x).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x).view(B, C, -1)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) #* self.scale # BHW, BHW
        x_w = x_w.view(B*H5*W5, B, H5*W5)
        x_w = torch.max(x_w, -1).values # BHW, B
        x_w = x_w.mean(-1)
        #x_w = torch.mean(x_w, -1).values # BHW
        x_w = x_w.view(B, -1) * self.scale # B, HW
        x_w = F.softmax(x_w, dim=-1) # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W
 
        x = x * x_w
        x = self.conv6(x)

        return x


class SGS(nn.Module):
    def __init__(self, channel_in, num_groups1=8, num_groups2=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(SGS, self).__init__()
        self.num_heads = num_heads
        self.num_groups1 = num_groups1
        self.num_groups2 = num_groups2
        head_dim = channel_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.normer = partial(nn.LayerNorm, eps=1e-6)(channel_in)

        self.qkv = nn.Linear(channel_in, channel_in * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channel_in, channel_in)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, recursive_index=False):
        batch_size, chl, hei, wid = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N, C = x.shape
        if recursive_index == False:
            num_groups = self.num_groups1
        else:
            num_groups = self.num_groups2
            if num_groups != 1:
                idx = torch.randperm(N)
                x = x[:,idx,:]
                inverse = torch.argsort(idx)
        qkv = self.qkv(x).reshape(B, num_groups, N // num_groups, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)  
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_groups, N // num_groups, C)
        x = x.permute(0, 3, 1, 2).reshape(B, C, N).transpose(1, 2)
        if recursive_index == True and num_groups != 1:
            x = x[:,inverse,:]
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hei, w=wid)
        return x
