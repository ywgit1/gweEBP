#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 31/03/2020
# @Author  : Fangliang Bai
# @File    : lcnn9_tri.py
# @Software: PyCharm
# @Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from xfr.models.lightcnn import LightCNN_9Layers
#from utils.utils import *

class Split(nn.Module):
    def __init__(self, split_size, dim):
        super(Split, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.split_size, self.dim)
    
class Max(nn.Module):
    def forward(self, inputs):
        return torch.max(*inputs)

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        self.mode = type
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
        self.split = Split(out_channels, 1)
        self.max = Max()
        self.indicator = None
            
    def forward(self, x):
        x = self.filter(x)
        #out = torch.split(x, self.out_channels, 1)
        out = self.split(x)
        if self.mode == 0:
            # out_tmp = torch.cat((out[0], out[1]), dim=0)
            # arg = torch.argmax(out_tmp, dim=0)
            # arg = arg.cpu()
            # # indicator = torch.cat((1-arg, arg), dim=0).reshape((2, -1))
            # sz = x.shape[1]
            # indexes = np.arange(0, sz)
            # half_siz = int(sz/2)
            # indicator = (1 - arg) * indexes[:half_siz] + arg * indexes[half_siz:]
            # self.indicator = torch.tensor(indicator, device=x.device, requires_grad=False).long()

            out_tmp = torch.cat((out[0], out[1]), dim=0)
            arg = torch.argmax(out_tmp, dim=0)
            self.indicator = arg
            
        out = self.max(out)
        return out
    
class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x
    
class network_9layers(nn.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x


class LCNN9EmbeddingNet(nn.Module):
    def __init__(self):
        super(LCNN9EmbeddingNet, self).__init__()
        # self.name = os.path.splitext(os.path.split(__file__)[1])[0]
        self.reset()
        # self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[8, 8], stride=[1, 1], padding=0)
        # self.feat_extract = nn.Conv2d(128, 256, kernel_size=[1, 1], stride=(1, 1))
        self.avgpool = self.features[-1]

    def forward(self, x):
        self.features.eval()
        # with torch.no_grad():
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1) 
        out = self.fc2(x)
        return out

    def get_embedding(self, x):
        self.features.eval()
        # with torch.no_grad():
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
        
    def reset(self):
        basenet = network_9layers()

        self.features_part1 = nn.Sequential(
                *basenet.features[0:5]
        )
        self.features_part2 = nn.Sequential(
                *basenet.features[5:]
        )
        modules = []
        def extract_module(layer, modules):
            for _, module in layer._modules.items():
                if 'group' in str(module) or 'mfm' in str(module):
                    extract_module(module, modules)
                else:
                    modules.append(module)
        extract_module(basenet.features, modules)
        self.features = nn.Sequential(*modules)        
        self.fc = basenet.fc1
        self.fc2 = basenet.fc2
        # fix_parameters(self.features_part1)
        # fix_parameters(self.features_part2)
        
    def classifier(self, x):
        # x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1) 
        out = self.fc2(x)
        return out


def get_model(lcnn9_pth=None):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = LCNN9EmbeddingNet()
    if lcnn9_pth is not None:
        state_dict = torch.load(lcnn9_pth)
        model.load_state_dict(state_dict['model_state_dict'], strict=False) # loads our own trained model
        model.features_part1 = None
        model.features_part2 = None        
    return model
