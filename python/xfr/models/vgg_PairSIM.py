# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:27:07 2020

@author: Sivapriyaa
"""

import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import torch.nn.functional as F

class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)  # Activation of last convolutional layer
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
#        return x38     # Actual Vggface1 returns x38
#        return x36  # use this for pre-trained model
        # return x33  # use this for pre-trained model   # accuarcy increases #******
#        return [x33, x30]    # For pretrained model and visualization
        return [x38,x30]    # For final embedding and last_conv_feature_map(visualization)
#        return x30  # For Visualization

def vgg16_preprocess():
    meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688], 
                         'std': [1, 1, 1], 
                         'imageSize': [224, 224, 3],
                         'multiplier': 255.0 }
    vgg_data_transform = {'img_resize': 256, 'crop_type': 0, # 0: no crop, 1: centre_crop, 2: random_crop
                          'random_flip': False,
                          'override_meta_imsize': False,
                          'to_grayscale': False}
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    resize = vgg_data_transform['img_resize']
    crop_type = vgg_data_transform['crop_type']
    override_meta_imsize = vgg_data_transform['override_meta_imsize']
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
        
    if vgg_data_transform['override_meta_imsize']:
        im_size = (vgg_data_transform['img_resize'], vgg_data_transform['img_resize'])
    
    transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    if vgg_data_transform['to_grayscale']:
        transform_list += [transforms.Grayscale()]
    if vgg_data_transform['random_flip']:
        transform_list += [transforms.RandomHorizontalFlip()]
    transform_list += [transforms.ToTensor()]
    transform_list += [lambda x: x * meta['multiplier']]
    transform_list.append(normalize) 
    transform_list += [lambda x: x.unsqueeze(0)]
    return transforms.Compose(transform_list)
    

class VggEmbeddingNet(nn.Module):
   
    def __init__(self, vgg_tri_2_pth=None):      
        super(VggEmbeddingNet, self).__init__()             
        self.name = os.path.splitext(os.path.split(__file__)[1])[0]     
        self.reset()
                
    def forward(self, x):        
        if type(x) in [tuple, list]:            
            x, labels = x
        
        x = self.basenet_part1(x)
        feat = self.basenet_part2(x)
            
        x1 = feat.view(feat.size()[0], -1) # Flatten
        emd = self.fc(x1)
        # emd = F.normalize(emd, p=2, dim=1)
        self.emd_norm = torch.norm(emd, p=2, dim=1).detach().clone()
        x = self.fc2(emd)
        
        return x, feat  # To be consistent with the solver class # unflattened feature map, embedding
       

    # def get_embedding(self, x):
    #     if type(x) in (tuple, list): # bypass labels if they are in the input list
    #         x = x[0]
    #     if self.training:
    #         self.eval()
    #     with torch.no_grad():
    #         x1 = self.basenet_part1(x)
    #         x2 = self.basenet_part2(x1)
    #         x3 = x2.view(x2.size()[0], -1)
    #         x4 = self.fc(x3)
    #     return [x4,x2]  # embedding, unflattened feature_map
    
    
    def reset(self):
        basenet = Vgg_face_dag()
        
        layers_part1 = []
        layers_part2 = []
                
        layers_part1.append(basenet.conv1_1)
        layers_part1.append(basenet.relu1_1)
        layers_part1.append(basenet.conv1_2)
        layers_part1.append(basenet.relu1_2)
        layers_part1.append(basenet.pool1)
        
        layers_part1.append(basenet.conv2_1)
        layers_part1.append(basenet.relu2_1)
        layers_part1.append(basenet.conv2_2)
        layers_part1.append(basenet.relu2_2)
        layers_part1.append(basenet.pool2)
        
        layers_part1.append(basenet.conv3_1)
        layers_part1.append(basenet.relu3_1)
        layers_part1.append(basenet.conv3_2)
        layers_part1.append(basenet.relu3_2)
        layers_part1.append(basenet.conv3_3)
        layers_part1.append(basenet.relu3_3)
        
        
        layers_part2.append(basenet.pool3)
        layers_part2.append(basenet.conv4_1)
        layers_part2.append(basenet.relu4_1)
        layers_part2.append(basenet.conv4_2)
        layers_part2.append(basenet.relu4_2)
        layers_part2.append(basenet.conv4_3)
        layers_part2.append(basenet.relu4_3)
        layers_part2.append(basenet.pool4)
        
        layers_part2.append(basenet.conv5_1)
        layers_part2.append(basenet.relu5_1)
        layers_part2.append(basenet.conv5_2)
        layers_part2.append(basenet.relu5_2)
        layers_part2.append(basenet.conv5_3)
        layers_part2.append(basenet.relu5_3)
        layers_part2.append(basenet.pool5)  
        
        self.basenet_part1 = nn.Sequential(*layers_part1)
        self.basenet_part2 = nn.Sequential(*layers_part2)

        self.dropout = nn.Dropout(p=0.5) 
        # self.basenet.fc6 = nn.Identity()
        # self.basenet.relu6 = nn.Identity()
        # self.basenet.dropout6 = nn.Identity()
        # self.basenet.fc7 = nn.Identity()
        # self.basenet.relu7 = nn.Identity()
        # self.basenet.dropout7 = nn.Identity()
        # self.basenet.fc8 = nn.Identity()
#        self.basenet.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3], 
#                                          padding=0, dilation=1, ceil_mode=False)
         # stride [3,3] is to reduce the number of input features to the fc layer.
        self.fc = nn.Linear(25088, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 2, bias=False)
       
        
def get_model(vgg_tri_2_pth=None):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = VggEmbeddingNet(vgg_tri_2_pth)
    if vgg_tri_2_pth is not None:
        state_dict = torch.load(vgg_tri_2_pth)
        model.load_state_dict(state_dict['model_state_dict'], strict=False) # loads our own trained model
    return model
