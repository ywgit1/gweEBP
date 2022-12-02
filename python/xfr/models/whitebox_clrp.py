# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import PIL.ImageFile

from xfr.models.lcnn9_tri import LCNN9EmbeddingNet

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import skimage
import skimage.filters
import pandas as pd
import pdb
import copy
import gc

from xfr.models.resnet import Bottleneck
from xfr.models.resnet import convert_resnet101v4_image
from xfr.models.lightcnn import lightcnn_preprocess, lightcnn_preprocess_fbai
import xfr.models.lightcnn
from xfr.models.vgg_tri_2_vis import vgg16_preprocess
from xfr import utils
# from xfr.models.clrp_lib import R, Nega_flag, linear_forward_hook, conv_forward_hook
# from xfr.models.clrp_lib import conv_backward_hook, maxpool_forward_hook, maxpool_backward_hook, \
#     FirstConv_backward_hook, linear_backward_hook, initialise_rel, return_rel


class WhiteboxNetwork(object):
    def __init__(self, net):
        """
        A WhiteboxNetwork() is the class wrapper for a torch network to be used with the Whitebox() class
        The input is a torch network which satisfies the assumptions outlined in the README
        """
        self.net = net
        self.net.eval()
        self.first_conv_visited = False

    def _layer_visitor(self, hooks, net=None):
        """Recursively assign forward hooks and pre-forward hooks for all layers within containers"""
        """Returns list in order of forward visit with {'name'n:, 'hook':h} dictionary"""
        """Signature of hook functions must match register_forward_hook and register_pre_forward_hook in torch docs"""
        layerlist = []
        self.first_conv_visited = False
        net = self.net if net is None else net
        for name, layer in net._modules.items():
            hooks_ = []
            if isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck):
                layerlist.append({'name': str(layer), 'hooks': [None]})
                layerlist = layerlist + self._layer_visitor(hooks, layer)
            else:
                if isinstance(layer, nn.Conv2d):
                    hooks_.append(layer.register_forward_hook(hooks['conv']['forward']))
                    if self.first_conv_visited:
                        hooks_.append(layer.register_backward_hook(hooks['conv']['backward']))
                    else:
                        hooks_.append(layer.register_backward_hook(hooks['first_conv']['backward']))
                        self.first_conv_visited = True
                    
                elif isinstance(layer, nn.MaxPool2d):
                    hooks_.append(layer.register_forward_hook(hooks['maxpool']['forward']))
                    hooks_.append(layer.register_backward_hook(hooks['maxpool']['backward']))
                    
                elif isinstance(layer, nn.Linear):
                    hooks_.append(layer.register_forward_hook(hooks['linear']['forward']))
                    hooks_.append(layer.register_backward_hook(hooks['linear']['backward']))
                    
                else:
                    hooks_.append(layer.register_backward_hook(hooks['relu']['backward']))
                    
                layerlist.append({'name': str(layer), 'hooks': hooks_})
            
        return layerlist

    def encode(self, x):
        """Given an Nx3xUxV input tensor x, return a D dimensional vector encoding of shape NxD, one per image"""
        raise

    def classify(self, x):
        """Given an Nx3xUxV input tensor x, and a network with C classes, return NxC pre-softmax classification output for the network"""
        raise

    def clear(self):
        """Clear gradients, multiple calls to backwards should not accumulate"""
        """This function should not need to be overloaded"""
        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def restore_emd_layer(self):
        raise
        
    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Given two D-dimensional encodings x_mate and x_nonmate, construct a 2xD classifier layer which will be output for classify() with C=2."""
        """Replace the output of the encoding layer with a 2xD fully connected layer which will compute the inner product of the two encodings and 
        the probe"""
        raise

    def num_classes(self):
        """Return the number of classes for the current network"""
        raise

    def preprocess(self, im):
        """Given a PIL image im, preprocess this image to return a tensor that can be suitably input to the network for forward()"""
        raise


# class WhiteboxSTResnet(WhiteboxNetwork):
#     def __init__(self, net):
#         """A subclass of WhiteboxNetwork() which implements the whitebox API for a resnet-101 topology"""
#         self.net = net
#         self.net.eval()

#     def set_triplet_classifier(self, x_mate, x_nonmate):
#         """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
#         self.net.fc2 = nn.Linear(512, 2, bias=False)
#         self.net.fc2.weight = nn.Parameter(torch.cat((x_mate, x_nonmate), dim=0))

#     def encode(self, x):
#         """Return 512d normalized forward encoding for input tensor x"""
#         return self.net.forward(x, mode='encode')

#     def classify(self, x):
#         return self.net.forward(x, mode='classify')

#     def num_classes(self):
#         return self.net.fc2.out_features

#     def preprocess(self, im):
#         """PIL image to input tensor"""
#         return convert_resnet101v4_image(im.resize((224, 224))).unsqueeze(0)


class WhiteboxLightCNN9(WhiteboxNetwork):
    def __init__(self, net:LCNN9EmbeddingNet):
        """A subclass of WhiteboxNetwork() which implements the whitebox API for a Light CNN Topology"""
        """https://github.com/AlfredXiangWu/LightCNN"""
        self.net = net
        self.net.eval()
        self.f_preprocess = lightcnn_preprocess_fbai()
        model_weights = list(self.net.parameters())
        self.fc_w = model_weights[-4]
        self.fc_b = model_weights[-3]
        
    def restore_emd_layer(self, device=None):
        from xfr.models.lightcnn import mfm
        if device is None:
            device = next(self.net.parameters()).device
        self.net.fc = mfm(8*8*128, 256, type=0).to(device)
        self.net.fc.filter.weight = nn.Parameter(self.fc_w)
        self.net.fc.filter.bias = nn.Parameter(self.fc_b)
        
    # def set_triplet_classifier(self, x_mate, x_nonmate, merge_layers=True):
    #     """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
    #     if not merge_layers:
    #         # <editor-fold desc="[+] Original implementation ..."> # not merge layer
    #         self.net.fc2 = nn.Linear(256, 2, bias=False)
    #         self.net.fc2.weight = nn.Parameter(torch.cat((x_mate, x_nonmate), dim=0) / self.net.emd_norm)
    #         # </editor-fold>
    #     else:
    #         # <editor-fold desc="[+] Our new implementation"> # merge layer
    #         ## FBai
    #         indicator = self.net.fc.indicator
    #         model_weights = list(self.net.parameters())
    #         fc_w = model_weights[-4]
    #         fc_b = model_weights[-3]
    
    #         r = torch.range(0,255)
    #         index = (1 - indicator) * r + indicator * (r + 256)
    #         index = index.long()
    #         passed_fc_w = fc_w[index]
    #         passed_fc_b = fc_b[index]
    
    #         embedding_weights = torch.cat((x_mate, x_nonmate), dim=0)
    #         new_fc_w = torch.matmul(embedding_weights, passed_fc_w) / self.net.emd_norm
    #         new_fc_b = torch.matmul(embedding_weights, passed_fc_b) / self.net.emd_norm
    #         self.net.fc = nn.Linear(8192, 2)
    #         self.net.fc.weight.data.copy_(new_fc_w.data)
    #         self.net.fc.bias.data.copy_(new_fc_b.data)
    #         # Modify fc2 layer
    #         self.net.fc2 = nn.Identity(2)
    #         # </editor-fold>

    def set_triplet_classifier(self, x_probe, x_mate, x_nonmate):
        # <editor-fold desc="[+] Original implementation ..."> # not merge layer
        device = next(self.net.parameters()).device
        self.net.fc2 = nn.Linear(256, 2, bias=False).to(device)
        self.net.fc2.weight.data.copy_(torch.cat( (x_mate, x_nonmate), dim=0) / self.net.emd_norm)
        
        # </editor-fold>
       
        indicator = self.net.fc.indicator
        fc_weight = self.net.fc.filter.weight
        fc_bias = self.net.fc.filter.bias
        r = torch.range(0, 255)
        index = (1 - indicator) * r + indicator * (r + 256)
        index = index.long()
        self.net.fc = nn.Linear(fc_weight.shape[1], 256, bias=True)
        self.net.fc.weight.data.copy_(fc_weight[index, :])
        self.net.fc.bias.data.copy_(fc_bias[index])

    def encode(self, x):
        """Return 512d normalized forward encoding for input tensor x"""
        p, features = self.net(x)
        features = F.normalize(features, p=2, dim=1)
        return features

    def classify(self, x):
        p, features = self.net(x)
        return p

    def num_classes(self):
        # return self.net.fc2.out_features
        return 2    # FBai

    def preprocess(self, im):
        """PIL image to input tensor"""
        return self.f_preprocess(im)

    def _layer_visitor(self, f_forward=None, f_preforward=None, net=None):
        """Recursively assign forward hooks and pre-forward hooks for all layers within containers"""
        """Returns list in order of forward visit with {'name'n:, 'hook':h} dictionary"""
        """Signature of hook functions must match register_forward_hook and register_pre_forward_hook in torch docs"""
        layerlist = []
        net = self.net if net is None else net
        for name, layer in net._modules.items():
            hooks = []
            if isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck) or isinstance(layer, xfr.models.lightcnn.mfm) or isinstance(layer,
                                                                                                                                             xfr.models.lightcnn.group) or isinstance(
                    layer, xfr.models.lightcnn.resblock):
                layerlist.append({'name': str(layer), 'hooks': [None]})
                layerlist = layerlist + self._layer_visitor(f_forward, f_preforward, layer)
            else:
                if f_forward is not None:
                    hooks.append(layer.register_forward_hook(f_forward))
                if f_preforward is not None:
                    hooks.append(layer.register_forward_pre_hook(f_preforward))
                layerlist.append({'name': str(layer), 'hooks': hooks})
        return layerlist


class WhiteboxVGG16(WhiteboxNetwork):
    def __init__(self, net):
        self.net = net
        self.net.eval()
        self.f_preprocess = vgg16_preprocess()
        model_weights = list(self.net.parameters())
        self.fc_w = model_weights[-3]
        self.fc_b = model_weights[-2]

    def restore_emd_layer(self, device=None):
        if device is None:
            device = next(self.net.parameters()).device
        self.net.fc = nn.Linear(25088, 1024).to(device)
        self.net.fc.weight.data.copy_(self.fc_w)
        self.net.fc.bias.data.copy_(self.fc_b)

    # def set_triplet_classifier(self, x_probe, x_mate, x_nonmate, merge_layers=False):
    #     if not merge_layers:
    #         # <editor-fold desc="[+] Original implementation ..."> # not merge layer
    #         self.net.fc2 = nn.Linear(1024, 2, bias=False)
    #         self.net.fc2.weight.data.copy_(torch.cat( (x_mate, x_nonmate), dim=0) / self.net.emd_norm)
    #         # </editor-fold>
    #     else:
    #         # <editor-fold desc="[+] Our new implementation"> # merge layer
    #         # indicator = self.net.fc.indicator.bool()
    #         embedding_weights = torch.cat((x_mate, x_nonmate), dim=0)
    #         new_fc_w = torch.matmul(embedding_weights, self.fc_w)
    #         new_fc_b = torch.matmul(embedding_weights, self.fc_b)
    #         self.net.fc = nn.Linear(25088, 2) # Caution: This will invalidate hooks for computing EBP! Any hook that has been attached to it must be reinstated.
    #         self.net.fc.weight = nn.Parameter(new_fc_w / self.net.emd_norm)
    #         self.net.fc.bias = nn.Parameter(new_fc_b / self.net.emd_norm)
    #         # Modify fc2 layer
    #         self.net.fc2 = nn.Identity(2)
    #         # </editor-fold>
    
    # def set_triplet_classifier(self, x_probe, x_mate, x_nonmate):
    #     # <editor-fold desc="[+] Original implementation ..."> # not merge layer
    #     self.indicator_pm = (x_probe * x_mate > 0).detach()
    #     self.indicator_p_neg = (x_probe < 0).detach()
    #     self.indicator_p_pos = torch.logical_not(self.indicator_p_neg)
    #     self.indicator_pm_flip = torch.logical_and(self.indicator_p_neg, self.indicator_pm)
    #     self.indicator_pn = (x_probe * x_nonmate > 0).detach()
    #     self.indicator_pn_flip = torch.logical_and(self.indicator_p_neg, self.indicator_pn)
        
    #     self.net.fc2 = nn.Linear(1024, 2, bias=False)
    #     self.net.fc2.weight.data.copy_(torch.cat( (x_mate, x_nonmate), dim=0) / self.net.emd_norm)
        
    #     x_mate_ = torch.zeros_like(x_mate)
    #     x_mate_[self.indicator_pm] = torch.abs(x_mate[self.indicator_pm])
    #     self.net.fc2.pm_pos_weight = torch.cat((x_mate_, x_nonmate), dim=0).detach() / self.net.emd_norm
    #     x_nonmate_ = torch.zeros_like(x_nonmate)
    #     x_nonmate_[self.indicator_pn] = torch.abs(x_nonmate[self.indicator_pn])
    #     self.net.fc2.pn_pos_weight = torch.cat((x_mate, x_nonmate_), dim=0).detach() / self.net.emd_norm
    #     self.net.fc2.reverse_act_sign = True
    #     # </editor-fold>
        
    #     fc_weight = self.net.fc.weight.data
    #     pm_pos_weight = torch.zeros_like(fc_weight)
    #     pm_pos_weight[self.indicator_p_pos.view(-1)] = F.relu(fc_weight[self.indicator_p_pos.view(-1)])
    #     pm_pos_weight[self.indicator_pm_flip.view(-1)] = F.relu(-fc_weight[self.indicator_pm_flip.view(-1)])
    #     self.net.fc.pm_pos_weight = pm_pos_weight
    #     pn_pos_weight = torch.zeros_like(fc_weight)
    #     pn_pos_weight[self.indicator_p_pos.view(-1)] = F.relu(fc_weight[self.indicator_p_pos.view(-1)])
    #     pn_pos_weight[self.indicator_pn_flip.view(-1)] = F.relu(-fc_weight[self.indicator_pn_flip.view(-1)])
    #     self.net.fc.pn_pos_weight = pn_pos_weight
        
    def set_triplet_classifier(self, x_probe, x_mate, x_nonmate):
        # <editor-fold desc="[+] Original implementation ..."> # not merge layer
        device = next(self.net.parameters()).device
        self.net.fc2 = nn.Linear(1024, 2, bias=False).to(device)
        self.net.fc2.weight.data.copy_(torch.cat( (x_mate, x_nonmate), dim=0) / self.net.emd_norm)
    

    def encode(self, x):
        """Return 512d normalized forward encoding for input tensor x"""
        p, features = self.net(x)
        features = F.normalize(features, p=2, dim=1)
        return features

    def classify(self, x):
        p, features = self.net(x)
        return p

    def num_classes(self):
        # return self.net.fc2.out_features
        return 2    # FBai

    def preprocess(self, im):
        """PIL image to input tensor"""
        return self.f_preprocess(im)
    
    
class Whitebox_senet50_256(WhiteboxNetwork):

    def __init__(self, net):
        """https://github.com/ox-vgg/vgg_face2"""
        self.net = net
        self.net.eval()
        self.fc1 = nn.Linear(256, 2, bias=False)

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.fc1.weight = nn.Parameter(torch.cat((x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 256d normalized forward encoding for input tensor x"""
        return self.net(x)[0]

    def classify(self, x):
        return self.fc1(self.net(x)[0])

    def num_classes(self):
        return 256 if self.fc1 is None else self.fc1.out_features

    def preprocess(self, img):
        """PIL image to input tensor"""
        """https://github.com/ox-vgg/vgg_face2/blob/master/standard_evaluation/pytorch_feature_extractor.py"""

        mean = (131.0912, 103.8827, 91.4953)

        short_size = 224.0
        crop_size = (224, 224, 3)
        im_shape = np.array(img.size)  # in the format of (width, height, *)
        img = img.convert('RGB')

        ratio = float(short_size) / np.min(im_shape)
        img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),  # width
                               int(np.ceil(im_shape[1] * ratio))),  # height
                         resample=PIL.Image.BILINEAR)

        x = np.array(img)  # image has been transposed into (height, width)
        newshape = x.shape[:2]
        h_start = (newshape[0] - crop_size[0]) // 2
        w_start = (newshape[1] - crop_size[1]) // 2
        x = x[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
        x = x - mean
        x = torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
        return x


class Whitebox_resnet50_128(WhiteboxNetwork):

    def __init__(self, net):
        """https://github.com/ox-vgg/vgg_face2"""
        self.net = net
        self.net.eval()
        self.fc1 = nn.Linear(128, 2, bias=False)

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.fc1.weight = nn.Parameter(torch.cat((x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 128d normalized forward encoding for input tensor x"""
        return self.net(x)[0]

    def classify(self, x):
        fc1_device = next(self.fc1.parameters()).device
        if fc1_device != x.device:
            self.fc1 = self.fc1.to(x.device)
        return self.fc1(self.net(x)[0])

    def num_classes(self):
        return 128 if self.fc1 is None else self.fc1.out_features

    def preprocess(self, img):
        """PIL image to input tensor"""
        """https://github.com/ox-vgg/vgg_face2/blob/master/standard_evaluation/pytorch_feature_extractor.py"""

        mean = (131.0912, 103.8827, 91.4953)

        short_size = 224.0
        crop_size = (224, 224, 3)
        im_shape = np.array(img.size)  # in the format of (width, height, *)
        img = img.convert('RGB')

        ratio = float(short_size) / np.min(im_shape)
        img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),  # width
                               int(np.ceil(im_shape[1] * ratio))),  # height
                         resample=PIL.Image.BILINEAR)

        x = np.array(img)  # image has been transposed into (height, width)
        newshape = x.shape[:2]
        h_start = (newshape[0] - crop_size[0]) // 2
        w_start = (newshape[1] - crop_size[1]) // 2
        x = x[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
        x = x - mean
        x = torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
        return x


class Whitebox(nn.Module):
    def __init__(self, net, device, eps=1E-16):
        """
        Net must be WhiteboxNetwork object.

        ebp_version=7: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=True, subtree_mode='all', do_mated_similarity_gating=True)
        ebp_version=8: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=False, subtree_mode='all', do_mated_similarity_gating=True)
        ebp_version=9: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=True, subtree_mode='all', do_mated_similarity_gating=False)
        ebp_version=10: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=True, subtree_mode='norelu',
        do_mated_similarity_gating=True)
        ebp_version=11: Whitebox(..., eps=1E-12, with_bias=False).weighted_subtree_ebp (..., do_max_subtree=True, subtree_mode='all',
        do_mated_similarity_gating=True)

        """
        super(Whitebox, self).__init__()
        assert (isinstance(net, WhiteboxNetwork))
        self.net = net
        self.layerlist = None

        # batch size is not applied to all functions, just embeddings
        self.batch_size = 32

        # Create layer visitor
        hooks = {'conv': {'forward': self.conv_forward_hook, 'backward': self.conv_backward_hook},
                 'first_conv': {'backward': self.FirstConv_backward_hook},
                 'linear': {'forward': self.linear_forward_hook, 'backward': self.linear_backward_hook},
                 'maxpool': {'forward': self.maxpool_forward_hook, 'backward': self.maxpool_backward_hook},
                 'relu': {'backward': self.relu_backward_hook}
                 }
        self.layerlist = self.net._layer_visitor(hooks)
        self.device = device
        self.eps = eps
        self.Nega_flag = False
        self.R = None
        self.R_list = []
        self.convert_saliency_uint8 = False
        
    def initialise_rel(self, R_, Nega_flag_):
        self.Nega_flag = Nega_flag_
        self.R = R_
        
    def return_rel(self):
        return self.R
    
    # hook function for Linear module
    def linear_forward_hook(self, module, input_, output): 
        if self._clrp_mode == 'disabled':
            pass
        module.input = input_[0]
        
    def linear_backward_hook(self, module, grad_input, grad_output):        
        if self._clrp_mode == 'disabled':
            pass
        W = module.parameters().__next__().clone()
        if self.Nega_flag == True:
            W *= torch.Tensor([-1]).to(self.device)
            self.Nega_flag = False
        
        V = torch.max(torch.zeros(W.t().shape).to(self.device), W.t())
        Z = torch.mm(module.input,V) + torch.Tensor([1e-9]).to(self.device)
        S = self.R / Z
        C = torch.mm(S, V.t())
        self.R = module.input * C
        self.R_list.append(self.R)
        
    # hook function for Conv module and Alpha_Beta rules for first Convolutional layer
    def conv_forward_hook(self, module, input_, output):   
        if self._clrp_mode == 'disable':
            pass
        module.input = input_[0]
    
    def Conv_single(self, input_, self_, R_in):    
        if self._clrp_mode == 'disable':
            pass
        with torch.enable_grad():
            
            input = input_.clone().detach().requires_grad_(True)
    
            V = torch.max(torch.zeros(self_.weight.shape).to(self.device), self_.weight)
    
            Z = F.conv2d(input, V, None, self_.stride,
                                self_.padding, self_.dilation, self_.groups) + torch.Tensor([1e-9]).to(self.device)
            S = R_in/Z
    
            Z.backward(S)
            return input * input.grad
        
        
    def conv_backward_hook(self, module, grad_input, grad_output):
        if self._clrp_mode == 'disable':
            pass
        
        self.R = self.Conv_single(module.input, module, self.R)
        self.R_list.append(self.R)
    
    def FirstConv_backward_hook(self, module, grad_input, grad_output):
        if self._clrp_mode == 'disable':
            pass
        
        with torch.enable_grad():
    
            VL = torch.max(torch.zeros(module.weight.shape).to(self.device), module.weight)
    
            VH = torch.min(torch.zeros(module.weight.shape).to(self.device), module.weight)
    
            Input = module.input.clone().detach().requires_grad_(True).to(self.device)
    
            Low = torch.tensor((torch.ones(module.input.shape) * torch.Tensor([-2.1179])).to(self.device), requires_grad=True)
    
            High = torch.tensor((torch.ones(module.input.shape) * torch.Tensor([2.64])).to(self.device), requires_grad=True)
    
            Z1 = F.conv2d(Input, module.weight, None, module.stride,
                                    module.padding, module.dilation, module.groups)
    
            Z2 = F.conv2d(Low, VL, None, module.stride,
                                   module.padding, module.dilation, module.groups)
    
            Z3 = F.conv2d(High, VH, None, module.stride,
                                    module.padding, module.dilation, module.groups)
    
            Z = Z1 - Z2 - Z3 + torch.Tensor([1e-9]).to(self.device)
            S = self.R / Z
    
            Z3.backward(S); Z1.backward(S); Z2.backward(S); 
    
            self.R = Input*Input.grad - Low*Low.grad - High*High.grad
    
        self.R_list.append(self.R)
    
    # hook function for Max Pooling module
    def maxpool_forward_hook(self, module, input, output):
        if self._clrp_mode == 'disable':
            pass
        module.input = input[0]
        module.output_shape = output.shape
        
    
    def maxPool_relprop(self, input, size=3, stride=1, padding=0, R_temp = None):
        out = F.max_pool2d(input, kernel_size=size, stride=1, padding=padding)
        mb,nx,wy,hy = out.shape
        
        R_ = torch.zeros(out.shape).to(self.device)
        R_[:,:,::stride,::stride] = R_temp
        
        mb,nx,wx,hx = input.shape
        Re = torch.zeros((mb,nx,wx+2*padding,hx+2*padding)).to(self.device)
    
        index = torch.zeros(mb, size*size, nx, wy, hy).to(self.device)
    
        for i in range(size):
            for j in range(size):
                index[:,i*size+j] = torch.eq(input[:,:,i:i+wy,j:j+hy], out).to(self.device, dtype=torch.float32)
                            
        redisR = (1.0/index.sum(dim=1)*R_).to(self.device)
        
        for i in range(size):
            for j in range(size):
                Re[:,:,i:i+wy,j:j+hy] += index[:, i*size+j] * redisR           
        return Re[:, :, padding : wx+padding, padding : hx+padding]
    
    
    def maxpool_backward_hook(self, module, grad_input, grad_output):
        if self._clrp_mode == 'disable':
            pass
        
        if len(module.output_shape) == 4 and self.R.dim() == 2: 
            self.R = self.R.reshape(module.output_shape)
        
        self.R = self.maxPool_relprop(module.input, size=module.kernel_size[0], 
                                 stride=module.stride[0], padding=module.padding, 
                                 R_temp=self.R)
        self.R_list.append(self.R)


    def relu_backward_hook(self, module, grad_input, grad_output):
        if self._clrp_mode == 'disable':
            pass
        self.R_list.append(self.R)      
        
    def _float32_to_uint8(self, img):
        # float32 [0,1] rescaled to [0,255] uint8
        return np.uint8(255 * ((img - np.min(img)) / (self.eps + (np.max(img) - np.min(img)))))

    def _scale_normalized(self, img):
        # float32 [0,1] 
        img = np.float32(img)
        return (img - np.min(img)) / (self.eps + (np.max(img) - np.min(img)))

    def _to_saliency(self, P, blur_radius=2):
        """Convert marginal winning probability (MWP) output from EBP to uint8 saliency map, with pooling, normalization and blurring"""
        img = P  # pooled over channels
        if self.convert_saliency_uint8:
            img = np.uint8(255 * ((img - np.min(img)) / (self.eps + (np.max(img) - np.min(img)))))  # normalized [0,255]
            img = np.array(PIL.Image.fromarray(img).filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius)))  # smoothed
            img = np.uint8(255 * ((img - np.min(img)) / (self.eps + (np.max(img) - np.min(img)))))  # renormalized [0,255]
        else:
            # version 6, avoid converting to 8 bit
            img = skimage.filters.gaussian(img, blur_radius)
            img = np.maximum(0, img)
            img /= (max(img.sum(), self.eps))
        return img


    def _clear(self):                                                                                                                                                    
        self.R_list = []
        gc.collect()

    def encode(self, x):
        """ Expose wbnet encode function.
        """
        return self.net.encode(x)

    def embeddings(self, images, norm=True):
        """ Calculate embeddings from numpy float images.

            A wrapper to help fit into existing API.
        """
        if isinstance(images, pd.DataFrame):
            imagesT = [self.convert_from_numpy(im)[0]
                       for im in utils.image_loader(images)]
        elif isinstance(images[0], torch.Tensor):
            assert images[0].ndim == 3  # [3, spatial dims]
            imagesT = images
        elif isinstance(images[0], np.ndarray):
            # currently only handle np arrays that are in network format
            # already
            assert images[0].shape[0] in (1, 3)  # grayscale or RGB
            imagesT = [torch.from_numpy(im).float() for im in images]
        else:
            imagesT = [self.convert_from_numpy(im)[0]
                       for im in utils.image_loader(images)]

        if not isinstance(imagesT, torch.Tensor):
            # imagesT = torch.cat(imagesT)
            imagesT = torch.stack(imagesT)

        batches = torch.split(imagesT, self.batch_size, dim=0)
        embeds = []
        for k, batch in enumerate(batches):
            batch = batch.to(next(self.net.net.parameters()).device)
            embeds.append(self.encode(batch).detach().cpu().numpy())
        embeds = np.concatenate(embeds)

        if norm:
            embeds = (
                    embeds.reshape((embeds.shape[0], -1)) /
                    np.linalg.norm(embeds.reshape((embeds.shape[0], -1)),
                                   axis=1, keepdims=True)
            ).reshape(embeds.shape)

        return embeds 
    
    def convert_from_numpy(self, img):
        """ Converts float RGB image (WxHx3) with range 0 to 1 or uint8 image
            to tensor (1x3XWxH).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255

        if img.max() > 1 + 1e-6 and img.min() > 0 - 1e-6:
            img = img / 255

        if img.max() > 1 + 1e-6 or img.min() < 0 - 1e-6:
            import pdb
            pdb.set_trace()
            img = (img - img.min()) / (img.max() - img.max() + 1e-6)

        # img = skimage.transform.resize(img, (224, 224), preserve_range=True)
        img = (img * 255).astype(np.uint8)
        img = PIL.Image.fromarray(img).convert('RGB')
        img = self.net.preprocess(img)
        return img

    def preprocess_loader(self, images, returnImageIndex=False, repeats=1):
        """ Iterates over tuple: (displayable image, tensor, fn)

            Tensor should have 3 dimensions (for a single image).

            Included to match snet interface.

            Preprocessing depends on specific network.
        """
        for im, fn in utils.image_loader(
                images,
                returnFileName=True,
                returnImageIndex=returnImageIndex,
                repeats=repeats,
        ):
            imT = self.convert_from_numpy(im)
            yield im, imT[0], fn


    # to create a saliency map of the target signal or the contrastive signal
    def _clrp(self, img_probe, target_class, device, contrastive_signal = False, CLRP='type1'):
        
        # initialize the relevance, i.e., the logit score
        Nega_flag_ = False
        x = img_probe.detach().clone()
        out = self.net.classify(x.requires_grad_(True))
        assert out[:, 0] > out[:, 1]
        if contrastive_signal == False:
            R = torch.zeros(out.shape).to(device)
            R[0, target_class] = out[0, target_class].detach()
        else:
            if CLRP == 'type1': 
                R = out.detach()
                R[0, target_class] = 0
            elif CLRP == 'type2':
                R = torch.zeros(out.shape).to(device)
                R[0, target_class] = out[0, target_class].detach()
                
            Nega_flag_ = True
    
        if Nega_flag_:
            self._clrp_mode = 'clrp'
        else:
            self._clrp_mode = 'lrp'
            
        # (Nega_flag_= False) for CLRP1 and (Nega_flag_= True) for CLRP2
        self.initialise_rel(torch.abs(R), Nega_flag_ = Nega_flag_)
    
        # propagate the relevance score back, which is computed in registered hook functions
        out.backward(out, retain_graph=False)
        R = self.return_rel().cpu().detach().numpy()
        
        return R
    
    def clrp(self, img_probe, target_class, device, contrastive_signal = False, CLRP='type1', k_R=-2):
        self._clear()
        self.net.net.zero_grad()
        self._clrp(img_probe, target_class, device, contrastive_signal=False)
        sm_lrp = self.R_list[k_R].cpu().detach().numpy()
        self._clear()
        self.net.net.zero_grad()
        self._clrp(img_probe, target_class, device, contrastive_signal=True, CLRP='type1')
        sm_contrast = self.R_list[k_R].cpu().detach().numpy()
        
        sm_lrp_ = sm_lrp/sm_lrp.sum()
        sm_contrast = sm_contrast/sm_contrast.sum()
        sm_clrp = sm_lrp_ - sm_contrast

        sm_clrp_ = (sm_clrp[0]/sm_clrp.max()).transpose((1, 2, 0))
        sm_clrp_ = np.maximum(0, sm_clrp_)
        sm_clrp_ = np.mean(sm_clrp_, axis=2)
        R = np.array(PIL.Image.fromarray(sm_clrp_).resize((img_probe.shape[3], img_probe.shape[2])))
        
        return self._to_saliency(R, blur_radius=2)