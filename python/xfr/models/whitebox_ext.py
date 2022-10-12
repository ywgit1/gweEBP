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


class WhiteboxNetwork(object):
    def __init__(self, net):
        """
        A WhiteboxNetwork() is the class wrapper for a torch network to be used with the Whitebox() class
        The input is a torch network which satisfies the assumptions outlined in the README
        """
        self.net = net
        self.net.eval()

    def _layer_visitor(self, f_forward=None, f_preforward=None, net=None):
        """Recursively assign forward hooks and pre-forward hooks for all layers within containers"""
        """Returns list in order of forward visit with {'name'n:, 'hook':h} dictionary"""
        """Signature of hook functions must match register_forward_hook and register_pre_forward_hook in torch docs"""
        layerlist = []
        net = self.net if net is None else net
        for name, layer in net._modules.items():
            hooks = []
            if isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck):
                layerlist.append({'name': str(layer), 'hooks': [None]})
                layerlist = layerlist + self._layer_visitor(f_forward, f_preforward, layer)
            else:
                if f_forward is not None:
                    hooks.append(layer.register_forward_hook(f_forward))
                if f_preforward is not None:
                    hooks.append(layer.register_forward_pre_hook(f_preforward))
                layerlist.append({'name': str(layer), 'hooks': hooks})
            if isinstance(layer, nn.BatchNorm2d):
                # layer.eval()
                # layer.track_running_stats = False  (do not disable, this screws things up)
                # layer.affine = False
                pass
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


class WhiteboxSTResnet(WhiteboxNetwork):
    def __init__(self, net):
        """A subclass of WhiteboxNetwork() which implements the whitebox API for a resnet-101 topology"""
        self.net = net
        self.net.eval()

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.net.fc2 = nn.Linear(512, 2, bias=False)
        self.net.fc2.weight = nn.Parameter(torch.cat((x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 512d normalized forward encoding for input tensor x"""
        return self.net.forward(x, mode='encode')

    def classify(self, x):
        return self.net.forward(x, mode='classify')

    def num_classes(self):
        return self.net.fc2.out_features

    def preprocess(self, im):
        """PIL image to input tensor"""
        return convert_resnet101v4_image(im.resize((224, 224))).unsqueeze(0)


class WhiteboxLightCNN(WhiteboxNetwork):
    def __init__(self, net):
        """A subclass of WhiteboxNetwork() which implements the whitebox API for a Light CNN Topology"""
        """https://github.com/AlfredXiangWu/LightCNN"""
        self.net = net
        self.net.eval()
        self.f_preprocess = lightcnn_preprocess()

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.net.fc2 = nn.Linear(256, 2, bias=False)
        self.net.fc2.weight = nn.Parameter(torch.cat( (x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 512d normalized forward encoding for input tensor x"""
        p, features = self.net(x)
        return features

    def classify(self, x):
        p, features = self.net(x)
        return p

    def num_classes(self):
        return self.net.fc2.out_features

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
        
    def restore_emd_layer(self):
        from xfr.models.lightcnn import mfm
        self.net.fc = mfm(8*8*128, 256, type=0)
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
        
        self.net.fc2 = nn.Linear(256, 2, bias=False)
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

    def restore_emd_layer(self):
        self.net.fc = nn.Linear(25088, 1024)
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
        
        self.net.fc2 = nn.Linear(1024, 2, bias=False)
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
    def __init__(self, net, ebp_version=None, with_bias=None, eps=1E-16,
                 ebp_subtree_mode='affineonly_with_prior'):
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

        self.eps = eps
        self.layerlist = None
        self.ebp_ver = ebp_version
        if self.ebp_ver is None:
            self.ebp_ver = 6  # set to the latest
        elif self.ebp_ver < 4:
            raise RuntimeError('ebp version, if set, must be at least 4')
        self.convert_saliency_uint8 = (self.ebp_ver != 6)
        if with_bias is not None:
            self._ebp_with_bias = with_bias
        else:
            self._ebp_with_bias = self.ebp_ver == 11

        self.dA = []  # layer activation gradient
        self.A = []  # layer activations
        self.X = []  # W^{+T}*A, equivalent to Apox
        self.P = []  # MWP layer outputs
        self.P_prior = []  # MWP layer priors
        self.P_layername = []  # MWP layer names in recursive layer order
        self.posA = [] # Positive activations of layers
        self.negA = [] # Negative activations of layers
        self.X_np = []
        self.X_pn = []
        self.X_pp = []
        self.X_nn = []
        self.gradlist = []

        # batch size is not applied to all functions, just embeddings
        self.batch_size = 32

        # Create layer visitor
        self._ebp_mode = 'disable'
        self.layerlist = self.net._layer_visitor(f_preforward=self._preforward_hook, f_forward=self._forward_hook)
        self._ebp_subtree_mode = ebp_subtree_mode
        self._ebp_ext_mode = None
        
        
    def _preforward_hook(self, module, x_input):
        if self._ebp_mode == 'activation':
            if hasattr(module, 'orig_weight') and module.orig_weight is not None:
                module.weight.data.copy_(module.orig_weight.data)  # Restore original weights
                module.orig_weight = None
            if hasattr(module, 'orig_bias') and module.orig_bias is not None:
                module.bias.data.copy_(module.orig_bias.data)  # Restore original bias
                module.orig_bias = None
            return None
        
        elif self._ebp_mode == 'filter_negative_activation': # For analyzing the significance of negative features
            if hasattr(module, 'orig_weight') and module.orig_weight is not None:
                module.weight.data.copy_(module.orig_weight.data)  # Restore original weights
                module.orig_weight = None
            if hasattr(module, 'orig_bias') and module.orig_bias is not None:
                module.bias.data.copy_(module.orig_bias.data)  # Restore original bias
                module.orig_bias = None 
            if x_input[0].ndim != 4 or \
                ('LightCNN9' in str(self.net) and x_input[0].shape[1] != 3 and x_input[0].shape[2] != 128) \
                or \
                ('VGG16' in str(self.net) and x_input[0].shape[1] != 3 and x_input[0].shape[2] != 224):# Exclude the input image
                x_input = tuple([F.relu(x) for x in x_input])
            return x_input
            
        elif self._ebp_mode == 'positive_activation':
            assert(len(self.A) > 0)  # must have called activation first
            #print(str(module))
            if hasattr(module, 'weight'):
                module.orig_weight = module.weight.detach().clone()
                if hasattr(module, 'pm_pos_weight'):
                    if self._ebp_mode2 == 'pm':
                        module.weight.data.copy_(module.pm_pos_weight.data)
                    elif self._ebp_mode2 == 'pn':
                        module.weight.data.copy_(module.pn_pos_weight.data)
                else:
                    module.pos_weight = F.relu(module.orig_weight.data) # W_{+}
                    module.weight.data.copy_( module.pos_weight.data )  # module backwards is on positive weights
            if hasattr(module, 'bias') and module.bias is not None:
                module.orig_bias = module.bias.detach().clone()
                module.bias.data.zero_() # The original bias should not take part in the EBP process
            # if self._ebp_with_bias and hasattr(module, 'bias') and module.bias is not None:
            #     module.orig_bias = module.bias.detach().clone()
            #     module.pos_bias = F.relu(module.orig_bias.data)
            #     module.bias.data.copy_( module.pos_bias.data )
            
            # Save layerwise positive activations (self.X = W^{+T}*A) in recursive layer visitor order
            self.X.append(tuple([F.relu(x).detach().clone() for x in x_input]))
            A = self.posA.pop(0)  # override forward input -> activation (A)
            self.posA.append(A) # Reappend for EBP
            return A # Replace the input with the non-negative activation
        
        elif self._ebp_mode == 'pos_act_pos_weight':
            assert(len(self.posA) > 0)  # must have called activation first
            #print(str(module))

            if hasattr(module, 'weight'):
                if not hasattr(module, 'orig_weight') or module.orig_weight is None:
                    module.orig_weight = module.weight.detach().clone()
                if not hasattr(module, 'pos_weight') or module.pos_weight is None:
                    module.pos_weight = F.relu(module.orig_weight.data) # W_{+}
                module.weight.data.copy_( module.pos_weight.data )  # module backwards is on positive weights
            if hasattr(module, 'bias') and module.bias is not None:
                if not hasattr(module, 'orig_bias') or module.orig_bias is None:
                    module.orig_bias = module.bias.detach().clone()
                module.bias.data.zero_() # The original bias should not take part in the EBP process
            
            # Save layerwise positive activations (self.X = W^{+T}*A^{+}) in recursive layer visitor order
            self.X.append(tuple([F.relu(x).detach().clone() for x in x_input]))
            posA = self.posA.pop(0)  # override forward input -> activation (A)
            self.posA.append(posA) # Reappend for EBP
            return posA # Replace the input with the non-negative activation
        
        elif self._ebp_mode == 'neg_act_pos_weight':
            assert(len(self.negA) > 0)
            if hasattr(module, 'weight'):
                if not hasattr(module, 'orig_weight') or module.orig_weight is None:
                    module.orig_weight = module.weight.detach().clone()
                if not hasattr(module, 'pos_weight') or module.pos_weight is None:
                    module.pos_weight = F.relu(module.orig_weight.data) # W_{+}
                module.weight.data.copy_( module.pos_weight.data )  # module backwards is on positive weights
            if hasattr(module, 'bias') and module.bias is not None:
                if not hasattr(module, 'orig_bias') or module.orig_bias is None:
                    module.orig_bias = module.bias.detach().clone()
                module.bias.data.zero_() # The original bias should not take part in the EBP process
            
            # Save layerwise negative activations (self.X = W^{+T}*A^{-T}) in recursive layer visitor order
            self.X_np.append(tuple([(-F.relu(-x)).detach().clone() for x in x_input]))
            negA = self.negA.pop(0)  # override forward input -> activation (A)
            self.negA.append(negA) # Reappend for EBP
            return negA # Replace the input with the negative activation            
        
        elif self._ebp_mode == 'pos_act_neg_weight':
            assert(len(self.posA) > 0)  # must have called activation first           
            if hasattr(module, 'weight'):
                if not hasattr(module, 'orig_weight') or module.orig_weight is None:
                    module.orig_weight = module.weight.detach().clone()
                if not hasattr(module, 'neg_weight') or module.neg_weight is None:
                    module.neg_weight = -F.relu(-1 * module.orig_weight.data) #torch.mul(module.orig_weight.data, module.orig_weight < 0) # W_{-}
                module.weight.data.copy_( module.neg_weight.data )  # module backwards is on positive weights
            if hasattr(module, 'bias') and module.bias is not None:
                if not hasattr(module, 'orig_bias') or module.orig_bias is None:
                    module.orig_bias = module.bias.detach().clone()
                module.bias.data.zero_() # The original bias should not take part in the EBP process
            
            # Save layerwise negative activations (self.X = W^{-T}*A^{+}) in recursive layer visitor order
            self.X_pn.append(tuple([(-F.relu(-x)).detach().clone() for x in x_input]))
            posA = self.posA.pop(0)  # override forward input -> activation (A)
            self.posA.append(posA) # Reappend for EBP
            return posA # Replace the input with the non-negative activation  

        elif self._ebp_mode == 'neg_act_neg_weight':
            assert(len(self.negA) > 0)
            if hasattr(module, 'weight'):
                if not hasattr(module, 'orig_weight') or module.orig_weight is None:
                    module.orig_weight = module.weight.detach().clone()
                if not hasattr(module, 'neg_weight') or module.neg_weight is None:
                    module.neg_weight = -F.relu(-1 * module.orig_weight.data) #torch.mul(module.orig_weight.data, module.orig_weight < 0) # W_{-}
                module.weight.data.copy_( module.neg_weight.data )  # module backwards is on positive weights
            if hasattr(module, 'bias') and module.bias is not None:
                if not hasattr(module, 'orig_bias') or module.orig_bias is None:
                    module.orig_bias = module.bias.detach().clone()
                module.bias.data.zero_() # The original bias should not take part in the EBP process
            
            # Save layerwise negative activations (self.X = W^{-T}*A^{-}) in recursive layer visitor order
            self.X_nn.append(tuple([F.relu(x).detach().clone() for x in x_input]))
            negA = self.negA.pop(0)  # override forward input -> activation (A)
            self.negA.append(negA) # Reappend for EBP
            return negA # Replace the input with the negative activation              
            
        elif self._ebp_mode in ['ebp', 'eebp']:
            assert(len(self.X) > 0 and len(self.posA) > 0 and len(self.X)==len(self.posA))  # must have called forward in activation and positive_activation mode first
            if hasattr(module, 'orig_weight') and module.orig_weight is not None:
                module.weight.data.copy_(module.orig_weight.data)  # restore weight
                module.orig_weight = None
            if hasattr(module, 'orig_bias') and module.orig_bias is not None:
                module.bias.data.copy_(module.orig_bias.data) # restore bias
                module.orig_bias = None
            # if self._ebp_with_bias and hasattr(module, 'orig_bias') and module.orig_bias is not None:
            #     module.bias.data.copy_(module.orig_bias.data)  # Restore original bias
            #     module.orig_bias = None
            return None
        
        elif self._ebp_mode == 'disable':
            if hasattr(module, 'orig_weight') and module.orig_weight is not None:
                module.weight.data.copy_(module.orig_weight.data)  # Restore original weights
                module.orig_weight = None
            if hasattr(module, 'orig_bias') and module.orig_bias is not None:
                module.bias.data.copy_(module.orig_bias.data) # restore bias
                module.orig_bias = None                
            # if self._ebp_with_bias and hasattr(module, 'orig_bias') and module.orig_bias is not None:
            #     module.bias.data.copy_(module.orig_bias.data)  # Restore original bias
            #     module.orig_bias = None
            return None
        else:
            raise ValueError('invalid mode "%s"' % self._ebp_mode)

    def _forward_hook(self, module, x_input, x_output):
        """Forward hook called after forward is called for each layer to save off layer inputs and gradients, or compute EBP"""
        if self._ebp_mode == 'activation':
            # Save layerwise activations (self.A) and layerwise gradients (self.dA) in recursive layer visitor order
            for x in x_input:
                def _savegrad(x):
                    self.dA.append(x)  # save layer gradient on backward

                x.register_hook(_savegrad)  # tensor hook to call _savegrad

            # if hasattr(module, 'reverse_act_sign'): # Flip the sign of negative activations which have positive contribution to the cosine similarity
            #     def reverse_sign(x):
            #         x_ = x.detach().clone()
            #         if self._ebp_mode2 == 'pm': # probe to mate
            #             mask = self.net.indicator_pm_flip[0]
            #             x_[:, mask] = -x_[:, mask]
            #         elif self._ebp_mode2 == 'pn': # probe to non-mate
            #             x_[:, self.net.indicator_pn_flip[0]] = -x_[:, self.net.indicator_pn_flip[0]]
            #         return x_
            #     if module.reverse_act_sign == True:
            #         self.A.append(tuple([F.relu(reverse_sign(x)) for x in x_input]))
            #     else:
            #         self.A.append(tuple([F.relu(x.detach().clone()) for x in x_input]))
            # else:
            #     self.A.append(tuple([F.relu(x.detach().clone()) for x in x_input]))  # save non-negative layer activation on forward
            # return None  # no change to forward output

            self.A.append(tuple([x.detach().clone() for x in x_input]))
            self.posA.append(tuple([F.relu(x.detach().clone()) for x in x_input]))
            self.negA.append(tuple([-F.relu(-1 * x).detach().clone() for x in x_input]))
            return None  # no change to forward output
        
        elif self._ebp_mode == 'filter_negative_activation':
            return None
            
        elif self._ebp_mode == 'positive_activation':
            return None
        
        elif self._ebp_mode == 'pos_act_pos_weight' or self._ebp_mode == 'pos_act_neg_weight' or \
            self._ebp_mode == 'neg_act_pos_weight' or self._ebp_mode == 'neg_act_neg_weight':
            return None

        elif self._ebp_mode == 'ebp':
            # Excitation backprop: https://arxiv.org/pdf/1608.00507.pdf, Algorithm 1, pg 9
            A = self.posA.pop(0)  # A_{n}: An input, pre-computed in "activation" mode
            X = self.X.pop(0)  # X_{n} = W^{+T}A_{n+1}, (Alg. 1, step 2), pre-computed in "positive activation" mode

            # Affine layers only
            if hasattr(module, 'pm_pos_weight'):
                module.orig_weight = module.weight.detach().clone()
                if self._ebp_mode2 == 'pm': # probe to mate  
                    module.weight.data.copy_(module.pm_pos_weight.data)
                elif self._ebp_mode2 == 'pn': # probe to non-mate
                    module.weight.data.copy_(module.pn_pos_weight.data)
            elif hasattr(module, 'pos_weight'):     
                # Step 1, W^{+}
                module.orig_weight = module.weight.detach().clone()
                module.weight.data.copy_(module.pos_weight.data)
            if hasattr(module, 'bias') and module.bias is not None: # Set bias to zero
                module.orig_bias = module.bias.detach().clone()
                module.bias.data.zero_()
            # if self._ebp_with_bias and hasattr(module, 'pos_bias'):
            #     module.orig_bias = module.bias.detach().clone()
            #     module.bias.data.copy_(module.pos_bias.data)

            for (g, a, x) in zip(x_input, A, X):
                assert (g.shape == a.shape and g.shape == x.shape)

                def _backward_ebp(z): # Backward hook for input tensors
                    # Tensor hooks are broken but "it's a feature", need operations in same scope to avoid memory leak
                    #   https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                    #   https://github.com/pytorch/pytorch/issues/12863
                    #   https://github.com/pytorch/pytorch/issues/25723

                    # Implement equation 10 (algorithm 1)
                    zh = F.relu(z) # Z is back-propagated from the input tensor of the n-th layer to the (n-1)-th layer
                    p = torch.mul(a, zh)  # step 5, closure (a): MWP for this layer: P_{n} = A_{n} * Z
                    p_prior = self.P_prior.pop(0) if len(self.P_prior) > 0 else None
                    if p_prior is not None:
                        p.data.copy_(p_prior)  # override with prior
                    self.P_layername.append(str(module))  # MWP layer, closure (self.P_layername)
                    self.P.append(p)  # marginal winning probability, closure (self.P)

                    # Subtree EBP modes (for analysis purposes)
                    if self._ebp_subtree_mode == 'affineonly':
                        # This mode sucks
                        if 'Conv' in str(module) or 'Linear' in str(module) or 'AvgPool' in str(module) or 'BatchNorm' in str(module):
                            y = torch.div(p, x + self.eps)  # step 3, closure (x): Y = P_{n} / (W^{+T} * A_{n+1}) for calculating MWP for the (n+1)-th layer
                            return y  # step 4: Y in (Z=W^{+}*Y) resulted from next (bottom) layer call to _backward_ebp, input to this lambda is Z
                        elif 'Sigmoid' in str(module) or 'ELU' in str(module) or 'Tanh' in str(module):
                            raise ValueError(
                                'layer "%s" is a special case (https://arxiv.org/pdf/1608.00507.pdf, eq 5), and is not yet supported' % str(module))
                        else:
                            return None # Pass "z"
                    elif self._ebp_subtree_mode == 'affineonly_with_prior':
                        zh = torch.mul(p_prior > 0, z) if p_prior is not None else zh
                        p = torch.mul(p_prior > 0, p) if p_prior is not None else p
                        if 'Conv' in str(module) or 'Linear' in str(module) or 'AvgPool' in str(module) or 'BatchNorm' in str(module):
                            y = torch.div(p, x + self.eps)  # step 3, closure (x)
                            return y  # step 4 (Z=W^{+}*Y), on next layer call to _backward_ebp, input to this lambda is Z
                        elif 'Sigmoid' in str(module) or 'ELU' in str(module) or 'Tanh' in str(module):
                            raise ValueError(
                                'layer "%s" is a special case (https://arxiv.org/pdf/1608.00507.pdf, eq 5), and is not yet supported' % str(module))
                        else:
                            return zh
                    elif self._ebp_subtree_mode == 'norelu': 
                        # This mode is necessary for visualization of other networks, without backprop -> inf
                        if ('MaxPool' in str(module) or 'ReLU' in str(module)) and p_prior is not None:
                            return None
                        elif 'Sigmoid' in str(module) or 'ELU' in str(module) or 'Tanh' in str(module):
                            raise ValueError(
                                'layer "%s" is a special case (https://arxiv.org/pdf/1608.00507.pdf, eq 5), and is not yet supported' % str(module))
                        else:
                            y = torch.div(p, x + self.eps)  # step 3, closure (x)
                            return y  # step 4 (Z=W^{+}*Y), on next layer call to _backward_ebp, input to this lambda is Z
                    elif self._ebp_subtree_mode == 'all':
                        # This mode is best for weighted subtree on STR-Janus network.  Why?
                        y = torch.div(p, x + self.eps)  # step 3, closure (x)
                        return y  # step 4 (Z=W^{+}*Y), on next layer call to _backward_ebp, input to this lambda is Z
                    else:
                        raise ValueError('Invalid subtree mode "%s"' % self._ebp_subtree_mode)

                g.register_hook(_backward_ebp)
            return None
            
        elif self._ebp_mode == 'eebp':
            # Excitation backprop: https://arxiv.org/pdf/1608.00507.pdf, Algorithm 1, pg 9
            A = self.A.pop(0)  # A_{n}: An input, pre-computed in "activation" mode
            self.A.append(A)
            X = self.X.pop(0)  # X_{n} = W^{+T}A_{n+1}, (Alg. 1, step 2), pre-computed in "positive activation" mode
            self.X.append(X)
            posA = self.posA.pop(0)
            self.posA.append(posA)
            negA = self.negA.pop(0)
            self.negA.append(negA)
            X_pn = self.X_pn.pop(0)
            self.X_pn.append(X_pn)
            X_np = self.X_np.pop(0)
            self.X_np.append(X_np)
            X_nn = self.X_nn.pop(0)
            self.X_nn.append(X_nn)
            dA = self.gradlist.pop(0)
            self.gradlist.append(dA)

            # Affine layers only
            if 'pos_weight' in self._ebp_ext_mode and hasattr(module, 'pos_weight'):     
                # Step 1, W^{+}
                module.orig_weight = module.weight.detach().clone()
                module.weight.data.copy_(module.pos_weight.data)
            elif 'neg_weight' in self._ebp_ext_mode and hasattr(module, 'neg_weight'):
                # Step 1, W^{-}
                module.orig_weight = module.weight.detach().clone()
                module.weight.data.copy_(module.neg_weight.data)
                
            if hasattr(module, 'bias') and module.bias is not None: # Set bias to zero
                module.orig_bias = module.bias.detach().clone()
                module.bias.data.zero_()
            # if self._ebp_with_bias and hasattr(module, 'pos_bias'):
            #     module.orig_bias = module.bias.detach().clone()
            #     module.bias.data.copy_(module.pos_bias.data)

            for (g, a, pa, na, x_pp, x_pn, x_np, x_nn) in zip(x_input, A, posA, negA, X, X_pn, X_np, X_nn):
                assert (g.shape == a.shape and g.shape == x_pp.shape)

                def _backward_eebp(z): # Backward hook for input tensors
                    # Tensor hooks are broken but "it's a feature", need operations in same scope to avoid memory leak
                    #   https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                    #   https://github.com/pytorch/pytorch/issues/12863
                    #   https://github.com/pytorch/pytorch/issues/25723

                    # Implement equation 10 (algorithm 1)
                    #zh = F.relu(z) # Z is back-propagated from the input tensor of the n-th layer to the (n-1)-th layer
                    if self._ebp_ext_mode == 'pos_act_pos_weight':
                        p = torch.mul(pa, z) # step 5, closure (a): MWP for this layer: P_{n} = A^{+}_{n} * Z 
                        if dA is not None:
                            p = p * F.relu(torch.mul(dA, a > 0))
                    elif self._ebp_ext_mode == 'pos_act_neg_weight':
                        p = torch.mul(pa, z)
                        if dA is not None:
                            p = p * F.relu(torch.mul(dA, a > 0))
                    elif self._ebp_ext_mode == 'neg_act_pos_weight':
                        p = torch.mul(na, z) # P^{-}_{n} = A^{-}_{n} * Z
                        if dA is not None:
                            p = p * F.relu(-torch.mul(dA, a < 0))
                    elif self._ebp_ext_mode == 'neg_act_neg_weight':
                        p = torch.mul(na, z)
                        if dA is not None:
                            p = p * F.relu(-torch.mul(dA, a < 0))                        
                    else:
                        raise ValueError('Unknown ebp_ext_mode %s' % self._ebp_ext_mode)
                    p = F.relu(p) # P must be non-negative
                    p_prior = None
                    if len(self.P_prior) > 0:
                        p_prior = self.P_prior.pop(0)
                        # self.P_prior.append(p_prior)
                    if p_prior is not None:
                        p.data.copy_(p_prior)  # override with prior
                    self.P_layername.append(str(module))  # MWP layer, closure (self.P_layername)
                    self.P.append(p)  # marginal winning probability, closure (self.P)

                    # Prepare Y = P / X for the next/bottom layer
                    if self._ebp_ext_mode in ['pos_act_pos_weight']:
                        # if 'Linear' in str(module) and 'in_features=256' in str(module): # debug 
                        #     x = x_pp# + x_nn # x > 0
                        # else:
                        x = x_pp + x_nn
                        #assert(torch.sum(x < 0) == 0)
                        x = x + self.eps
                        p_ = torch.mul(p, a > 0) # P(A^{+})
                        # if dA is not None:
                        #     dA_ = F.relu(dA)
                        #     p_ = p_ * dA_
                    elif self._ebp_ext_mode in ['neg_act_neg_weight']:
                        x = x_nn + x_pp
                        x = x + self.eps
                        p_ = torch.mul(p, a > 0)
                        # if dA is not None:
                        #     dA_ = F.relu(dA)
                        #     p_ = p_ * dA_
                    elif self._ebp_ext_mode in ['pos_act_neg_weight']:#, 'neg_act_pos_weight']:
                        # if 'Linear' in str(module):#debug
                        #     x = x_pn # + x_np # x < 0
                        # else:
                        x = x_pn + x_np
                        #assert(torch.sum(x > 0) == 0)
                        x = x - self.eps
                        p_ = torch.mul(p, a < 0) # P(A^{-})
                        # if dA is not None:
                        #     dA_ = F.relu(-dA)
                        #     p_ = p_ * dA_
                    elif self._ebp_ext_mode in ['neg_act_pos_weight']:
                        x = x_np + x_pn
                        x = x - self.eps
                        p_ = torch.mul(p, a < 0)
                        # if dA is not None:
                        #     dA_ = F.relu(-dA)
                        #     p_ = p_ * dA_
                    else:
                        raise ValueError('Unknown ebp_ext_mode %s' % self._ebp_ext_mode)
                        
                    # Subtree EBP modes (for analysis purposes)
                    if self._ebp_subtree_mode == 'affineonly':
                        # This mode sucks
                        if 'Conv' in str(module) or 'Linear' in str(module) or \
                            'AvgPool' in str(module) or 'BatchNorm' in str(module):
                            y = torch.div(p_, x)  # step 3, closure (x): Y = P^{+/-}_{n} / (W^{+/-T} * A^{+/-}_{n+1}) for calculating MWP for the (n+1)-th layer
                            return y  # step 4: Y in (Z=W^{+}*Y) resulted from next (bottom) layer call to _backward_ebp, input to this lambda is Z
                        # elif 'Sigmoid' in str(module) or 'ELU' in str(module) or 'Tanh' in str(module):
                        #     raise ValueError(
                        #         'layer "%s" is a special case (https://arxiv.org/pdf/1608.00507.pdf, eq 5), and is not yet supported' % str(module))
                        # elif 'ReLU' in str(module):
                        #     z = torch.mul(z, a > 0)
                        #     return z
                        else:
                            return None # Pass "z"
                    else:
                        raise ValueError('Invalid subtree mode "%s"' % self._ebp_subtree_mode)

                g.register_hook(_backward_eebp)
            return None
        
        elif self._ebp_mode == 'disable':
            return None
        else:
            raise ValueError('Invalid mode "%s"' % self._ebp_mode)

    def _float32_to_uint8(self, img):
        # float32 [0,1] rescaled to [0,255] uint8
        return np.uint8(255 * ((img - np.min(img)) / (self.eps + (np.max(img) - np.min(img)))))

    def _scale_normalized(self, img):
        # float32 [0,1] 
        img = np.float32(img)
        return (img - np.min(img)) / (self.eps + (np.max(img) - np.min(img)))

    def _mwp_to_saliency(self, P, blur_radius=2):
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

    # def _layers(self):
    #     """Random input EBP just to get layer order on forward.  This sets hooks."""
    #     img = self._float32_to_uint8(np.random.rand( 224,224,3 ))
    #     P = torch.zeros( (1, self.net.num_classes()), dtype=torch.float32 )
    #     x = self.net.preprocess(PIL.Image.fromarray(img))
    #     if next(self.net.net.parameters()).is_cuda:
    #         the following only works for single-gpu systems:
    #         P = P.cuda()
    #         x = x.cuda()
    #     self.ebp(x, P)
    #     P_layername = self.P_layername
    #     self._clear()
    #     return P_layername

    def _clear(self):
        (self.P, self.P_layername, self.dA, self.A, self.X) = ([], [], [], [], [])
        (self.posA, self.negA, self.X_np, self.X_pn, self.X_nn) = ([], [], [], [], [])
        self.net.clear()

    def contrastive_eebp(self, img_probe, k_poschannel, k_negchannel, K=None, k_mwp=-2):
        assert(k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert(k_negchannel >= 0 and k_negchannel < self.net.num_classes())

        # K = None
        # if 'LightCNN9' in str(self.net):
        #     K = 12
        #     k_mwp = 10
        # elif 'VGG16' in str(self.net):
        #     K = 9
        #     k_mwp = -1
            
        # Mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.eebp(img_probe, P0, mwp=False, K=K, k_mwp=k_mwp)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.eebp(img_probe, P0, mwp=False, K=K, k_mwp=k_mwp)
        P_nonmate = self.P
        
        # Contrastive EBP
        mwp_mate = P_mate[k_mwp] / torch.sum(P_mate[k_mwp]) 
        mwp_nonmate = P_nonmate[k_mwp] / torch.sum(P_nonmate[k_mwp]) 
        mwp_contrastive = np.squeeze(np.sum(F.relu(mwp_mate - mwp_nonmate).detach().cpu().numpy(), axis=1).astype(np.float32))  # pool over channels
        mwp_contrastive = np.array(PIL.Image.fromarray(mwp_contrastive).resize((img_probe.shape[3], img_probe.shape[2])))
        return self._mwp_to_saliency(mwp_contrastive)                                                                                                                                                     
    

    def truncated_contrastive_eebp(self, img_probe, k_poschannel, k_negchannel, percentile=20, K=None, k_mwp=-2):
        """Truncated contrastive excitation backprop"""
        assert (k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert (k_negchannel >= 0 and k_negchannel < self.net.num_classes())

        # K = None
        # if 'LightCNN9' in str(self.net):
        #     K = 12
        #     k_mwp = 10
        # elif 'VGG16' in str(self.net):
        #     K = 9
        #     k_mwp = -1
            
        # Mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.eebp(img_probe, P0, mwp=False, K=K, k_mwp=k_mwp)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.eebp(img_probe, P0, mwp=False, K=K, k_mwp=k_mwp)
        P_nonmate = self.P

        # Truncated contrastive EBP
        mwp_mate = P_mate[k_mwp] / torch.sum(P_mate[k_mwp])# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer
        mwp_nonmate = P_nonmate[k_mwp] / torch.sum(P_nonmate[k_mwp])# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer
        (mwp_sorted, mwp_sorted_indices) = torch.sort(torch.flatten(mwp_mate.clone()))  # ascending
        mwp_sorted_cumsum = torch.cumsum(mwp_sorted, 0)  # for percentile
        percentile_mask = torch.zeros(mwp_sorted.shape)
        percentile_mask[mwp_sorted_indices] = (mwp_sorted_cumsum >= (percentile / 100.0) * mwp_sorted_cumsum[-1]).type(torch.FloatTensor)
        percentile_mask = percentile_mask.reshape(mwp_mate.shape)
        percentile_mask = percentile_mask.to(img_probe.device)
        mwp_mate = torch.mul(percentile_mask, mwp_mate)        
        mwp_nonmate = torch.mul(percentile_mask, mwp_nonmate)
        mwp_mate = mwp_mate / torch.sum(mwp_mate)# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer
        mwp_nonmate = mwp_nonmate / torch.sum(mwp_nonmate)# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer        
        tcebp = F.relu(mwp_mate - mwp_nonmate)
        mwp_truncated_contrastive = np.squeeze(np.sum(tcebp.detach().cpu().numpy(), axis=1).astype(np.float32))  # pool over channels
        return self._mwp_to_saliency(mwp_truncated_contrastive)

    
    def eebp(self, x, Pn, mwp=False, K=None, k_mwp=-2):
        """Excitation backprop: forward operation to compute activations (An, Xn) and backward to compute Pn following equation (10)"""

        # Pre-processing
        x = x.detach().clone()  # if we do not clone, then the backward graph grows
        self._clear()  # if we do do not clear, then forward will accumulate self.A and self.dA
        gc.collect()

        # Forward activations
        self._ebp_mode = 'activation'
        y = self.net.classify(x.requires_grad_(True)).detach().cpu().numpy() # Obtain non-negative activations in a forward pass A_{n}
        assert np.all(y[:, 0] > y[:, 1])
        
        self._ebp_mode = 'pos_act_pos_weight' 
        _ = self.net.classify(x) # Get W^{+}. Generate X^{+}_{1} for each layer in a forward pass as: X = W^{+T} * A^{+}_{n}
        self._ebp_mode = 'neg_act_pos_weight' 
        _ = self.net.classify(x) # Get W^{+}. Generate X^{-}_{1} for each layer in a forward pass as: X = W^{+T} * A^{-}_{n}
        self._ebp_mode = 'pos_act_neg_weight' 
        _ = self.net.classify(x) # Get W^{-}. Generate X^{-}_{2} for each layer in a forward pass as: X = W^{-T} * A^{+}_{n}
        self._ebp_mode = 'neg_act_neg_weight' 
        _ = self.net.classify(x) # Get W^{-}. Generate X^{+}_{2} for each layer in a forward pass as: X = W^{-T} * A^{-}_{n} 

        # K = None
        # if 'LightCNN9' in str(self.net):
        #     K = 12
        #     k_mwp = 10
        # elif 'VGG16' in str(self.net):
        #     K = 9
        #     k_mwp = -1
            
        # E2BP
        self._ebp_mode = 'eebp'
        self._ebp_ext_mode = 'pos_act_pos_weight'
        self.P_layername, self.P = [], []
        Xn = self.net.classify(x)
        Xn.backward(Pn, retain_graph=False) # In a backward pass, generate MWP P_{n} = A_{n} * Z
        if K is None: K = len(self.P_layername)
        
        P = []
        p_0, _ = self.layerwise_eebp(x, 0, K, None, Pn) 
        P.append(p_0)
        p_prior = p_0
        k = 0
        
        while k < K:
            p_next, k_next = self.layerwise_eebp(x, k, K, p_prior, Pn)
            if p_next is not None:
                assert(len(p_next) > 0)
                for p in p_next: P.append(p)
                p_prior = p_next[-1]
                k = k_next[-1]
            else:
                break
        
        self.P = P
        
        P = np.squeeze(np.sum(P[k_mwp].detach().cpu().numpy(), axis=1)).astype(np.float32)  # pool over channels
        self._ebp_mode = 'disable'
        
        # P = np.array(PIL.Image.fromarray(P).resize((224, 224)))
        
        # Marginal winning probability or saliency map
        P = self._mwp_to_saliency(P) if not mwp else P
        return P
    
    def layerwise_eebp(self, x, k_layer, K, p_prior, Pn):
        """Layerwise excitation backprop for extended EBP"""
        """For a given layer, set a prior and back propagate from there"""
        # assert (k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        # Pn = torch.zeros((1, self.net.num_classes()))
        # Pn[0][k_poschannel] = 1.0  # one-hot
        Pn = Pn.to(x.device)
        
        if p_prior is None:
            k_ret = k_layer
        else:
            k_ret = k_layer + 1
            while k_ret < K:
                if 'MaxPool' in self.P_layername[k_ret] or 'Split' in self.P_layername[k_ret] or 'ReLU' in self.P_layername[k_ret]:
                    k_ret += 1
                else:
                    break
        
        P_ret = None
        if k_ret < K:
            self._ebp_mode = 'eebp'
            self._ebp_ext_mode = 'pos_act_pos_weight'
            self.P_layername, self.P = [], []
            self.P_prior = [None for x in range(K)]  # all other layers propagate
            self.P_prior[k_layer] = p_prior
            Xn = self.net.classify(x.requires_grad_(True))
            self.net.net.zero_grad()
            Xn.backward(Pn, retain_graph=False) # In a backward pass, generate MWP P_{n} = A_{n} * Z
            P_ret = [self.P[kk] for kk in range(k_layer + 1, k_ret + 1)] if p_prior is not None else self.P[k_ret]
            
        # if k_ret == 1:
            self._ebp_ext_mode = 'pos_act_neg_weight'
            self.P_layername, self.P = [], []
            self.P_prior = [None for x in range(K)]  # all other layers propagate
            self.P_prior[k_layer] = p_prior
            Xn = self.net.classify(x)
            self.net.net.zero_grad()
            Xn.backward(Pn, retain_graph=False) # In a backward pass, generate MWP P_{n} = A_{n} * Z
            P_ret = [P_ret[kk - k_layer - 1] + self.P[kk] for kk in range(k_layer + 1, k_ret + 1)] if p_prior is not None else P_ret + self.P[k_ret]
            
            self._ebp_ext_mode = 'neg_act_pos_weight'
            self.P_layername, self.P = [], []
            self.P_prior = [None for x in range(K)]  # all other layers propagate
            self.P_prior[k_layer] = p_prior
            Xn = self.net.classify(x)
            self.net.net.zero_grad()
            Xn.backward(Pn, retain_graph=False) # In a backward pass, generate MWP P_{n} = A_{n} * Z
            P_ret = [P_ret[kk - k_layer - 1] + self.P[kk] for kk in range(k_layer + 1, k_ret + 1)] if p_prior is not None else P_ret + self.P[k_ret]
            
        # if k_ret == 0:
            self._ebp_ext_mode = 'neg_act_neg_weight'
            self.P_layername, self.P = [], []
            self.P_prior = [None for x in range(K)]  # all other layers propagate
            self.P_prior[k_layer] = p_prior
            Xn = self.net.classify(x)
            self.net.net.zero_grad()
            Xn.backward(Pn, retain_graph=False) # In a backward pass, generate MWP P_{n} = A_{n} * Z
            P_ret = [P_ret[kk - k_layer - 1] + self.P[kk] for kk in range(k_layer + 1, k_ret + 1)] if p_prior is not None else P_ret + self.P[k_ret]
        
        k_ret = list(range(k_layer + 1, k_ret + 1)) if p_prior is not None else k_ret
        
        return P_ret, k_ret
    
    
    def ebp(self, x, Pn, mwp=False, k_mwp=-2):
        """Excitation backprop: forward operation to compute activations (An, Xn) and backward to compute Pn following equation (10)"""

        # Pre-processing
        x = x.detach().clone()  # if we do not clone, then the backward graph grows
        self._clear()  # if we do do not clear, then forward will accumulate self.A and self.dA
        gc.collect()

        # Forward activations
        self._ebp_mode = 'activation'
        y = self.net.classify(x.requires_grad_(True)).detach().cpu().numpy() # Obtain non-negative activations in a forward pass A_{n}
        assert np.all(y[:, 0] > y[:, 1])
        
        self._ebp_mode = 'positive_activation' 
        _ = self.net.classify(x.requires_grad_(True)) # Get W^{+}. Generate X for each layer in a forward pass as: X = W^{*T} * A_{n}
            
        self._ebp_mode = 'ebp'
        Xn = self.net.classify(x.requires_grad_(True))
        Xn.backward(Pn, retain_graph=True) # In a backward pass, generate MWP P_{n} = A_{n} * Z
        P = np.squeeze(np.sum(self.P[k_mwp].detach().cpu().numpy(), axis=1)).astype(np.float32)  # pool over channels
        self._ebp_mode = 'disable'
        
        # P = np.array(PIL.Image.fromarray(P).resize((224, 224)))
        
        # Marginal winning probability or saliency map
        P = self._mwp_to_saliency(P) if not mwp else P
        return P

    def contrastive_ebp(self, img_probe, k_poschannel, k_negchannel, k_mwp=-2):
        """Contrastive excitation backprop"""
        assert(k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert(k_negchannel >= 0 and k_negchannel < self.net.num_classes())

        # Mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self._ebp_mode2 = 'pm' # probe is matched to mate
        self.ebp(img_probe, P0, k_mwp)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self._ebp_mode2 = 'pn' # probe is matched to non-mate
        self.ebp(img_probe, P0, k_mwp)
        P_nonmate = self.P
        
        # Contrastive EBP
        # if 'LightCNN9' in str(self.net):
        #     k_mwp = 10
        # elif 'VGG16' in str(self.net):
        #     k_mwp = 8
        self._ebp_mode2 = None
        mwp_mate = P_mate[k_mwp] / torch.sum(P_mate[k_mwp]) # -2: MWP of the first conv. layer, 2: MWP of the last conv. layer
        mwp_nonmate = P_nonmate[k_mwp] / torch.sum(P_nonmate[k_mwp]) # -2: MWP of the first conv. layer, 2: MWP of the last conv. layer
        mwp_contrastive = np.squeeze(np.sum(F.relu(mwp_mate - mwp_nonmate).detach().cpu().numpy(), axis=1).astype(np.float32))  # pool over channels
        mwp_contrastive = np.array(PIL.Image.fromarray(mwp_contrastive).resize((img_probe.shape[3], img_probe.shape[2])))
        return self._mwp_to_saliency(mwp_contrastive)

    def truncated_contrastive_ebp(self, img_probe, k_poschannel, k_negchannel, percentile=20, k_mwp=-2):
        """Truncated contrastive excitation backprop"""
        assert (k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert (k_negchannel >= 0 and k_negchannel < self.net.num_classes())

        # Mated EBP
        P0 = torch.zeros((1, self.net.num_classes()));
        P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self._ebp_mode2 = 'pm'
        self.ebp(img_probe, P0, k_mwp)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros((1, self.net.num_classes()));
        P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self._ebp_mode2 = 'pn'
        self.ebp(img_probe, P0, k_mwp)
        P_nonmate = self.P

        # Truncated contrastive EBP
        # if 'LightCNN9' in str(self.net):
        #     k_mwp = 10
        # elif 'VGG16' in str(self.net):
        #     k_mwp = 8
        mwp_mate = P_mate[k_mwp] / torch.sum(P_mate[k_mwp])# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer
        mwp_nonmate = P_nonmate[k_mwp] / torch.sum(P_nonmate[k_mwp])# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer

        (mwp_sorted, mwp_sorted_indices) = torch.sort(torch.flatten(mwp_mate.clone()))  # ascending
        mwp_sorted_cumsum = torch.cumsum(mwp_sorted, 0)  # for percentile
        percentile_mask = torch.zeros(mwp_sorted.shape)
        percentile_mask[mwp_sorted_indices] = (mwp_sorted_cumsum >= (percentile / 100.0) * mwp_sorted_cumsum[-1]).type(torch.FloatTensor)
        percentile_mask = percentile_mask.reshape(mwp_mate.shape)
        percentile_mask = percentile_mask.to(img_probe.device)
        mwp_mate = torch.mul(percentile_mask, mwp_mate)        
        mwp_nonmate = torch.mul(percentile_mask, mwp_nonmate)
        mwp_mate = mwp_mate / torch.sum(mwp_mate)# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer
        mwp_nonmate = mwp_nonmate / torch.sum(mwp_nonmate)# -2: MWP of the first conv. layer, 2: MWP of the last conv. layer        
        tcebp = F.relu(mwp_mate - mwp_nonmate)
        mwp_truncated_contrastive = np.squeeze(np.sum(tcebp.detach().cpu().numpy(), axis=1).astype(np.float32))  # pool over channels
        return self._mwp_to_saliency(mwp_truncated_contrastive)


    def layerwise_ebp(self, img_probe, k_layer, mode='argmax', k_element=None, k_poschannel=0, mwp=True):
        """Layerwise excitation backprop"""
        """For a given layer, select the starting node according to a provided element or the provided mode"""
        assert (k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        P0 = torch.zeros((1, self.net.num_classes()));
        P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_mate = self.P

        self.P_prior = [None for x in self.P]  # all other layers propagate
        if mode == 'argmax':
            self.P_prior[k_layer] = torch.mul(P_mate[k_layer], 1.0 - torch.ne(P_mate[k_layer], torch.max(P_mate[k_layer])).type(torch.FloatTensor))
        elif mode == 'elementwise':
            assert (k_element is not None)
            P = (0 * (P_mate[k_layer].detach().clone())).flatten()
            P[k_element] = P_mate[k_layer].flatten()[k_element]
            self.P_prior[k_layer] = P.reshape(P_mate[k_layer].shape)
        else:
            raise ValueError('invalid layerwise EBP mode "%s"' % mode)

        return self.ebp(img_probe, 0.0 * P0, mwp=mwp)

    def layerwise_contrastive_ebp(self, img_probe, k_poschannel, k_negchannel, k_layer, mode='copy', percentile=80, k_element=None, gradlayer=None,
                                  mwp=False):
        """Layerwise contrastive excitation backprop"""

        import warnings
        warnings.warn("layerwise_contrastive_ebp is deprecated, use weighted_subtree_ebp instead")

        # Mated EBP
        assert (k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert (k_negchannel >= 0 and k_negchannel < self.net.num_classes())
        P0 = torch.zeros((1, self.net.num_classes()));
        P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros((1, self.net.num_classes()));
        P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_nonmate = self.P

        # Contrastive EBP
        self.P_prior = [None for x in self.P]  # all other layers propagate
        if mode == 'copy':
            # Pn is replaced with contrastive difference at layer k
            self.P_prior[k_layer] = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
        elif mode == 'mean':
            # Pn is replaced with mean of contrastive difference and EBP
            self.P_prior[k_layer] = 0.5 * (P_mate[k_layer] + (F.relu(P_mate[k_layer] - P_nonmate[k_layer])))
        elif mode == 'product':
            # Product of EBP and contrast
            self.P_prior[k_layer] = torch.sqrt(
                torch.mul(P_mate[k_layer].type(torch.DoubleTensor), F.relu(P_mate[k_layer] - P_nonmate[k_layer]).type(torch.DoubleTensor))).type(
                torch.FloatTensor)
        elif mode == 'argmax':
            Pn_prior = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            self.P_prior[k_layer] = torch.mul(Pn_prior, 1.0 - torch.ne(Pn_prior, torch.max(Pn_prior)).type(torch.FloatTensor))
        elif mode == 'argmax_product':
            Pn_prior = torch.sqrt(
                torch.mul(P_mate[k_layer].type(torch.DoubleTensor), F.relu(P_mate[k_layer] - P_nonmate[k_layer]).type(torch.DoubleTensor))).type(
                torch.FloatTensor)
            self.P_prior[k_layer] = torch.mul(Pn_prior, 1.0 - torch.ne(Pn_prior, torch.max(Pn_prior)).type(torch.FloatTensor))
        elif mode == 'percentile' or mode == 'percentile_argmax':
            assert (percentile >= 0 and percentile <= 100)
            Pn = P_mate[k_layer]
            Pn_prior = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            (Pn_sorted, Pn_sorted_indices) = torch.sort(torch.flatten(Pn.clone()))  # ascending
            Pn_sorted_cumsum = torch.cumsum(Pn_sorted, 0)  # for percentile
            Pn_mask = torch.zeros(Pn_sorted.shape)
            Pn_mask[Pn_sorted_indices] = (Pn_sorted_cumsum >= (percentile / 100.0) * Pn_sorted_cumsum[-1]).type(torch.FloatTensor)
            Pn_mask = Pn_mask.reshape(Pn.shape)
            self.P_prior[k_layer] = torch.mul(Pn_mask, Pn_prior.type(torch.FloatTensor)).clone()
            if mode == 'percentile_argmax':
                Pn = self.P_prior[k_layer]
                Pn_argmax = torch.mul(Pn, 1.0 - torch.ne(Pn, torch.max(Pn)).type(torch.FloatTensor))
                self.P_prior[k_layer] = Pn_argmax
        elif mode == 'elementwise':
            assert (gradlayer[k_layer].shape == P_mate[k_layer].shape)
            C = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            P = (0 * C.detach().clone()).flatten()
            P[k_element] = C.flatten()[k_element]
            self.P_prior[k_layer] = P.reshape(C.shape)
        else:
            raise ValueError('unknown contrastive ebp mode "%s"' % mode)

        return self.ebp(img_probe, 0.0 * P0, mwp=mwp)


    def weighted_subtree_ebp(self, img_probe, k_poschannel, k_negchannel, topk=1, verbose=True, do_max_subtree=False, do_mated_similarity_gating=True,
                             subtree_mode='norelu', do_mwp_to_saliency=True):
        """Weighted subtree EBP"""

        # Forward and backward to save triplet loss data gradients in layer visitor order
        self._ebp_subtree_mode = subtree_mode
        self._ebp_mode = 'activation'
        x = img_probe.detach().clone()  # if we do not clone, then the backward graph grows for some reason
        y = self.net.classify(x.requires_grad_(True))
        y0 = torch.tensor([0]).to(y.device)
        # y1 = torch.tensor([1]).cuda() if next(self.net.net.parameters()).is_cuda else torch.tensor([1])
        # F.cross_entropy(y, y0).backward(retain_graph=True)  # Binary classes = {mated, nonmated}
        # gradlist = self.dA
        self._clear()

        # Forward and backward to save mate and non-mate data gradients in layer visitor order
        if not do_mated_similarity_gating:
            y = self.net.classify(x.requires_grad_(True))
            F.cross_entropy(y, y0).backward(retain_graph=True)
            gradlist_ce = self.dA
            self._clear()

        y[0][0].backward(retain_graph=True)
        # gradlist_mated = copy.deepcopy(self.dA)  # TESTING deepcopy
        gradlist_mated = self.dA
        self._clear()

        y[0][1].backward(retain_graph=True)
        # gradlist_nonmated = copy.deepcopy(self.dA)  # TESTING deepcopy
        gradlist_nonmated = self.dA
        self._clear()

        # Select subtrees using loss weighting
        P_img = []
        P_subtree = []
        P_subtree_idx = []
        n_layers = len(gradlist_mated)
        assert len(gradlist_mated) == len(gradlist_nonmated)
        for k in range(0, n_layers - 1):  # not including image layer
            # Given loss function L, a positive gradient (dL/dx) means that x is DECREASED to DECREASE the loss
            # Given loss function L, a negative gradient (-dL/dx) means that x is INCREASED to DECREASE the loss (more excitory -> reduces loss)
            if do_mated_similarity_gating:
                # Mated similarity  must increase (positive gradient), non-mated similarity decrease (negative gradient, more excitory)
                p = torch.max(torch.mul(gradlist_mated[k] >= 0, -gradlist_nonmated[k]))
                k_argmax = torch.argmax(torch.mul(gradlist_mated[k] >= 0, -gradlist_nonmated[k]))
            else:
                # Triplet ebp loss must decrease (negative gradient, more excitory), non-mated similarity decrease (negative gradient, more excitory)
                p = torch.max(torch.mul(gradlist_ce[k] < 0, -gradlist_nonmated[k]))
                k_argmax = torch.argmax(torch.mul(gradlist_ce[k] < 0, -gradlist_nonmated[k]))
            P_subtree.append(float(p))
            P_subtree_idx.append(k_argmax)
        k_subtree = np.argsort(np.array(P_subtree))  # ascending, one per layer

        # Generate layerwise EBP for each selected subtree
        for k in k_subtree:
            P_img.append(self.layerwise_ebp(x, k_layer=k, k_poschannel=k_poschannel, k_element=P_subtree_idx[k], mode='elementwise'))
            if verbose:
                print('[weighted_subtree_ebp][%d]: layername=%s, grad=%f' % (k, self.P_layername[k], P_subtree[k]))

        # Merge MWP from each subtree, weighting by convex combination of subtrees, weights proportional to loss gradient
        k_valid = [np.max(P) > 0 for P in P_img]
        k_subtree_valid = [k for (k, v) in zip(k_subtree, k_valid) if v == True and k != 1][-topk:]  # FIXME: k==1 is for STR-Janus Multiply() layer
        if len(k_subtree_valid) == 0:
            # assert(len(k_subtree_valid)>0)  # Should never be empty
            raise RuntimeError(
                    'Failed to calculate valid subtrees. The ebp subtree mode '
                    '(%s) may not support by this type of network. You may want '
                    'to try the "affineonly_with_prior" ebp subtree mode.' %
                    self._ebp_subtree_mode
            )
        P_img_valid = [p for (p, k, v) in zip(P_img, k_subtree, k_valid) if v == True and k != 1][-topk:]
        P_subtree_valid = [P_subtree[k] for k in k_subtree_valid]
        P_subtree_valid_norm = self._scale_normalized(P_subtree_valid) if not np.sum(self._scale_normalized(P_subtree_valid)) == 0 else np.ones_like(
            P_subtree_valid)
        if do_max_subtree:
            smap = np.max(np.dstack([float(p_subtree_valid_norm) * np.array(P) * (1.0 / (np.max(P) + 1E-12)) for (p_subtree_valid_norm, P) in
                                     zip(P_subtree_valid_norm, P_img_valid)]), axis=2)
        else:
            if len(P_subtree_valid_norm) > 0:
                smap = np.sum(np.dstack([float(p_subtree_valid_norm) * np.array(P) * (1.0 / (np.max(P) + 1E-12)) for (p_subtree_valid_norm, P) in
                                         zip(P_subtree_valid_norm, P_img_valid)]), axis=2)
            else:
                smap = 0 * P_img[0]  # empty saliency map

        # Generate output saliency map
        if self.convert_saliency_uint8:
            smap = self._float32_to_uint8(smap)
        else:
            smap /= max(smap.sum(), self.eps)

        return (
            self._mwp_to_saliency(smap) if do_mwp_to_saliency else smap,
            [self._mwp_to_saliency(P) if do_mwp_to_saliency else P for P in P_img_valid],
            P_subtree_valid,
            k_subtree_valid)


    def gradient_weighted_ecEBP(self, img_probe, x_mate, x_nonmate, k_poschannel, k_negchannel, K=None, k_mwp=-2):
        assert(k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert(k_negchannel >= 0 and k_negchannel < self.net.num_classes())
            
        self._ebp_mode = 'activation'
        x = img_probe.detach().clone()  # if we do not clone, then the backward graph grows for some reason
        emd = self.net.encode(x.requires_grad_(True))
        x_mate, x_nonmate = x_mate.to(x.device), x_nonmate.to(x.device)
        triplet_gain = ((emd * x_mate).sum(1) - (emd * x_nonmate).sum(1)).mean()
        triplet_gain.backward(retain_graph=False)
        self.dA.reverse()
        self.gradlist = [None for x in range(len(self.dA))]
        # self.gradlist[-1] = self.dA[-1]
        # self.gradlist[-2] = self.dA[-2]
        # self.gradlist[-3] = self.dA[-3]
        # self.gradlist[len(self.dA) - k_mwp - 1] = self.dA[len(self.dA) - k_mwp - 1]
        self.gradlist = self.dA
        self._clear()
        
        # Mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.eebp(img_probe, P0, mwp=False, K=K, k_mwp=k_mwp)
        P_mate = self.P

        # Non-mated EBP
        # triplet_loss = ((emd * x_nonmate).sum(1) - (emd * x_mate).sum(1)).mean()
        # triplet_loss.backward(retain_graph=False)
        self.gradlist = [-grad if grad is not None else grad for grad in self.gradlist]
        self._clear()
        
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.eebp(img_probe, P0, mwp=False, K=K, k_mwp=k_mwp)
        P_nonmate = self.P
        
        # Contrastive EBP
        mwp_mate = P_mate[k_mwp] / torch.sum(P_mate[k_mwp]) 
        mwp_nonmate = P_nonmate[k_mwp] / torch.sum(P_nonmate[k_mwp]) 
        mwp_contrastive = np.squeeze(np.sum(F.relu(mwp_mate - mwp_nonmate).detach().cpu().numpy(), axis=1).astype(np.float32))  # pool over channels
        mwp_contrastive = np.array(PIL.Image.fromarray(mwp_contrastive).resize((img_probe.shape[3], img_probe.shape[2])))
        return self._mwp_to_saliency(mwp_contrastive)                                                                                                                                                     


    def ebp_subtree_mode(self):
        return self._ebp_subtree_mode

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
