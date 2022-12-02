#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 30/11/2022
# @Author  : Yan Wang
# @File    : lcnn9_agf.py
# @Software: PyCharm
# @Description:
"""

import torch
import torch.nn as torch_nn
from xfr.models.AGF import layers as nn
import torch.nn.functional as F

#from xfr.models.lightcnn import LightCNN_9Layers
#from utils.utils import *

class Split(nn.AGFPropSimple):
    def __init__(self, split_size, dim):
        super(Split, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.split_size, self.dim)
    
    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        # Gradient
        if type(cam) not in (tuple, list):
            raise ValueError('Expect cam to be a list or tuple')
        cam = torch.cat(cam, dim=1)
        Y = self.forward(self.X)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, self.X, S)
        
        return cam, grad_out    

class Max(nn.AGFPropSimple):
    def forward(self, inputs):
        return torch.max(*inputs)
    
class mfm(torch_nn.Module):
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

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        cam, grad_outputs = self.max.AGF(cam, grad_outputs)
        cam, grad_outputs = self.split.AGF(cam, grad_outputs)
        cam, grad_outputs = self.filter.AGF(cam, grad_outputs)
        return cam, grad_outputs
        
class group(nn.AGFProp):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        cam, grad_outputs = self.conv.AGF(cam, grad_outputs)
        cam, grad_outputs = self.conv_a.AGF(cam, grad_outputs)
        return cam, grad_outputs
    
class network_9layers(torch_nn.Module):
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

class LCNN9EmbeddingNet(torch_nn.Module):
    def __init__(self):
        super(LCNN9EmbeddingNet, self).__init__()
        # self.name = os.path.splitext(os.path.split(__file__)[1])[0]
        self.reset()
        # self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[8, 8], stride=[1, 1], padding=0)
        # self.feat_extract = nn.Conv2d(128, 256, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, x):
        self.features_part1.eval()
        # with torch.no_grad():
        x = self.features_part1(x)

        self.features_part2.eval()
        # with torch.no_grad():
        x = self.features_part2(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        self.emd_norm = torch.norm(x, p=2, dim=1).detach().clone()
        # Do not normalize the embedding for EBP as the normalization should not
        # be considered in the back-propagation. Instead, the embedding will be scaled 
        # by the normalization factor in fc2.
        # x = F.normalize(x, p=2, dim=1) 
        out = self.fc2(x)
        return out, x

    def AGF(self, **kwargs):
        cams, grad_outputs = self.classifier.AGF(**kwargs)
        cam_list = list(reversed(cams))
        cam = cams[-1].reshape_as(self.maxpool.Y)
        grad_outputs = (grad_outputs[0].reshape_as(self.maxpool.Y),)
        cam, grad_outputs = self.maxpool.AGF(cam, grad_outputs, **kwargs)
        cam_list = [cam] + cam_list
        cams, grad_outputs = self.features_part2.AGF(cam, grad_outputs, **kwargs)
        cam_list = list(reversed(cams)) + cam_list
        cams, grad_outputs = self.features_part1.AGF(cams[-1], grad_outputs, **kwargs)
        cam_list = list(reversed(cams)) + cam_list

        for i in range(len(cam_list)):
            cam = cam_list[i] / nn.minmax_dims(cam_list[i], 'max')
            if len(cam.shape) == 4:
                cam = cam.sum(1, keepdim=True)
            cam_list[i] = cam
            
        return cam_list
    
    def reset(self):
        basenet = network_9layers()

        self.features_part1 = nn.Sequential(
                *basenet.features[0:5]
        )
        self.features_part2 = nn.Sequential(
                *basenet.features[5:-1]
        )
        self.maxpool = basenet.features[-1]
        self.fc = basenet.fc1
        self.fc2 = basenet.fc2
        # fix_parameters(self.features_part1)
        # fix_parameters(self.features_part2)
        self.reset_classifier()

    def reset_classifier(self):
        self.classifier = nn.Sequential(self.fc, self.fc2)

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
    return model
