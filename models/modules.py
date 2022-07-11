import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from config import Config


config = Config()


class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(64)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x


class CoAttLayer(nn.Module):
    def __init__(self, channel_in=512):
        super(CoAttLayer, self).__init__()

        self.all_attention = eval(Config().relation_module + '(channel_in)')
        self.conv_output = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.conv_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.fc_transform = nn.Linear(channel_in, channel_in)

        for layer in [self.conv_output, self.conv_transform, self.fc_transform]:
            weight_init.c2_msra_fill(layer)
    
    def forward(self, x5):
        if self.training:
            f_begin = 0
            f_end = int(x5.shape[0] / 2)
            s_begin = f_end
            s_end = int(x5.shape[0])

            x5_1 = x5[f_begin: f_end]
            x5_2 = x5[s_begin: s_end]

            x5_new_1 = self.all_attention(x5_1)
            x5_new_2 = self.all_attention(x5_2)

            x5_1_proto = torch.mean(x5_new_1, (0, 2, 3), True).view(1, -1)
            x5_1_proto = x5_1_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            x5_2_proto = torch.mean(x5_new_2, (0, 2, 3), True).view(1, -1)
            x5_2_proto = x5_2_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            x5_11 = x5_1 * x5_1_proto
            x5_22 = x5_2 * x5_2_proto
            weighted_x5 = torch.cat([x5_11, x5_22], dim=0)

            x5_12 = x5_1 * x5_2_proto
            x5_21 = x5_2 * x5_1_proto
            neg_x5 = torch.cat([x5_12, x5_21], dim=0)
        else:

            x5_new = self.all_attention(x5)
            x5_proto = torch.mean(x5_new, (0, 2, 3), True).view(1, -1)
            x5_proto = x5_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            weighted_x5 = x5 * x5_proto #* cweight
            neg_x5 = None
        return weighted_x5, neg_x5


class GAM(nn.Module):
    def __init__(self, channel_in=512):

        super(GAM, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x5):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x5.size()

        x_query = self.query_transform(x5).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)

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
 
        x5 = x5 * x_w
        x5 = self.conv6(x5)

        return x5
