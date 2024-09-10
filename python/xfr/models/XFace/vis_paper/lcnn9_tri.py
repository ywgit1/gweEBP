#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 31/03/2020
# @Author  : Fangliang Bai
# @File    : lcnn9_tri.py
# @Software: PyCharm
# @Description:
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from vis_paper.lightcnn import LightCNN_9Layers
#from utils.utils import *


class LCNN9EmbeddingNet(nn.Module):
    def __init__(self):
        super(LCNN9EmbeddingNet, self).__init__()
        # self.name = os.path.splitext(os.path.split(__file__)[1])[0]
        self.reset()
        self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[8, 8], stride=[1, 1], padding=0)
        # self.feat_extract = nn.Conv2d(128, 256, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, x):
        self.features_part1.eval()
        # with torch.no_grad():
        x = self.features_part1(x)

        self.features_part2.eval()
        # with torch.no_grad():
        x = self.features_part2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=1)  # Yan
        self.emd_norm = torch.norm(x, p=2, dim=1).detach().clone()  # Yan sign flip
        out = self.fc2(x)
        return out, x

    def reset(self):
        basenet = LightCNN_9Layers()

        self.features_part1 = nn.Sequential(
                *basenet.features[0:5]
        )
        self.features_part2 = nn.Sequential(
                *basenet.features[5:]
        )
        self.fc = basenet.fc1
        self.fc2 = basenet.fc2
        # fix_parameters(self.features_part1)
        # fix_parameters(self.features_part2)


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


def get_embedding(img: np.array, model: nn.Module):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # model = get_model("/media/kent/DISK2/E2ID/Vis_Orig_LCNN9/trained_models/"
    #                   "lcnn9_log_results_Wed_06Oct2021_1535_Malta_SetA_anchor_sketch_type/ckt/lcnn9_tri_Wed_06Oct2021_173415_epoch30.pth")
    #
    # model.to(device)
    model.eval()
    
    class C(object):
        meta = {'mean': [0],
                'std': [1],
                'imageSize': [128, 128, 3],
                'multiplier': 1.0}
        data_transform = {'img_resize': 144, 'crop_type': 0,
                          'random_flip': False,
                          'override_meta_imsize': False,
                          'to_grayscale': True}
    
    def compose_transforms(meta, resize, to_grayscale, crop_type, override_meta_imsize, random_flip):
        """Compose preprocessing transforms for VGGFace model
    
        The imported models use a range of different preprocessing options,
        depending on how they were originally trained. Models trained in MatConvNet
        typically require input images that have been scaled to [0,255], rather
        than the [0,1] range favoured by PyTorch.
    
        Args:
            meta (dict): model preprocessing requirements
            resize (int) [256]: resize the input image to this size
            center_crop (bool) [True]: whether to center crop the image
            override_meta_imsize (bool) [False]: if true, use the value of `resize`
               to select the image input size, rather than the properties contained
               in meta (this option only applies when center cropping is not used.
    
        Return:
            (transforms.Compose): Composition of preprocessing transforms
        """
        normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
        im_size = meta['imageSize']
        assert im_size[0] == im_size[1], 'expected square image size'
        if crop_type == 1:
            transform_list = [transforms.Resize(resize),
                              transforms.CenterCrop(size=(im_size[0], im_size[1]))]
        elif crop_type == 2:
            transform_list = [transforms.Resize(resize),
                              transforms.RandomCrop(size=(im_size[0], im_size[1]))]
        else:
            if override_meta_imsize:
                im_size = (resize, resize)
            transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
        if to_grayscale:
            transform_list += [transforms.Grayscale()]
        if random_flip:
            transform_list += [transforms.RandomHorizontalFlip()]
        transform_list += [transforms.ToTensor()]
        transform_list += [lambda x: x * meta['multiplier']]
        transform_list.append(normalize)
        return transforms.Compose(transform_list)
    
    def _get_data_transforms():
        return compose_transforms(C.meta, resize=C.data_transform['img_resize'],
                                  to_grayscale=C.data_transform['to_grayscale'],
                                  crop_type=C.data_transform['crop_type'],
                                  override_meta_imsize=C.data_transform['override_meta_imsize'],
                                  random_flip=C.data_transform['random_flip'])
    
    from PIL import Image
    if len(img.shape) == 3:
        img = (img * 255).astype(np.uint8)
        qimg = _get_data_transforms()(Image.fromarray(img))
        qimg = qimg.unsqueeze(0)
        qimg_t = qimg.to(device)
        emd_query = model(qimg_t)
        emd_query = emd_query[1].detach().cpu()
        emd_query = emd_query.squeeze(0).numpy()
        return emd_query
    elif len(img.shape) == 4:
        emb_list = []
        for i in range(img.shape[0]):
            img_i = (img[i] * 255).astype(np.uint8)
            qimg_i = _get_data_transforms()(Image.fromarray(img_i))
            qimg_i = qimg_i.unsqueeze(0)
            qimg_t_i = qimg_i.to(device)
            emd_query_i = model(qimg_t_i)
            emd_query_i = emd_query_i[1].detach().cpu()
            emd_query_i = emd_query_i.squeeze(0).numpy()
            emb_list.append(emd_query_i)
            # stack the embeddings
        return np.stack(emb_list)
    
    
