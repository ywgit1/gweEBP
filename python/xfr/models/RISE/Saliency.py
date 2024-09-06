# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:10:51 2024

@author: Sivapriyaa

Randomized Image Sampling for Explanations (RISE)
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import *
from explanations import RISE, corrRISE
from PIL import Image

cudnn.benchmark= True

args=Dummy()

# Number of workers to load data
args.workers = 8
# Directory with images split into class folders.
# Since we don't use ground truth labels for saliency all images can be 
# moved to one class folder.
# args.datadir = '/scratch2/Datasets/imagenet/ILSVRC2012_val_folders/'
# # Sets the range of images to be explained for dataloader.
# args.range = range(95, 105)
# Size of imput images.
args.input_size = (224, 224)
# args.input_size = (256, 256)
# Size of batches for GPU. 
# Use maximum number that the GPU allows.
# args.gpu_batch = 250
args.gpu_batch = 32

""" Prepare data """
# dataset = datasets.ImageFolder(args.datadir, preprocess)

# This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=1, shuffle=False,
#     num_workers=args.workers, pin_memory=True, sampler=RangeSampler(args.range))

# print('Found {: >5} images belonging to {} classes.'.format(len(dataset), len(dataset.classes)))
# print('      {: >5} images will be explained.'.format(len(data_loader) * data_loader.batch_size))


""" Load black-box model """
# Load black box model for explanations
model = models.resnet50(True)
# model = nn.Sequential(model, nn.Softmax(dim=1)) # output from the last fc layer (classification layer) for RISE
model=nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer for corrRISE

model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False
    
# # To use multiple GPUs
model = nn.DataParallel(model)


""" Create Explainer Instance """
# explainer = RISE(model, args.input_size, args.gpu_batch)
explainer = corrRISE(model, args.input_size, args.gpu_batch)
# Generate masks for RISE or use the saved ones.
maskspath = 'masks.npy'
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
    # explainer.generate_masks(N=100, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')
    
"""Explaining one instance """
def example_RISE(img, top_k=3):
    saliency = explainer(img.cuda()).cpu().numpy() # Calculates saliency maps for all the 1000 classes
    p, c = torch.topk(model(img.cuda()), k=top_k)  #torch.topk returns values and indices
    p, c = p[0], c[0] 
    
    plt.figure(figsize=(10, 5*top_k))
    for k in range(top_k):
        plt.subplot(top_k, 2, 2*k+1)
        plt.axis('off')
        plt.title('{:.2f}% {}'.format(100*p[k], get_class_name(c[k])))
        tensor_imshow(img[0])

        plt.subplot(top_k, 2, 2*k+2)
        plt.axis('off')
        plt.title(get_class_name(c[k]))
        tensor_imshow(img[0])
        sal = saliency[c[k]] # Extracts the top 3 saliency maps
        plt.imshow(sal, cmap='jet', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()

def example_corrRISE(img1, img2,top_k=1):
    # saliency = explainer(img1.cuda(),img2.cuda()).cpu().numpy()
    saliency = explainer(img1.cuda(),img2.cuda())
    saliency1=saliency[0]
    saliency2=saliency[1]

    #Subplot 1
    plt.subplot(2, 2, 1)
    tensor_imshow(img1[0])
    #Subplot 2
    plt.subplot(2, 2, 2)
    tensor_imshow(img1[0])
    plt.imshow(saliency1, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    #Subplot 3
    plt.subplot(2, 2, 3)
    tensor_imshow(img2[0])
    #Subplot 4
    plt.subplot(2, 2, 4)
    tensor_imshow(img2[0])
    plt.imshow(saliency2, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.show()
    
def deprecated_example_corrRISE(img1, img2,top_k=1): # Doesn't make sense
    saliency1 = explainer(img1.cuda()).cpu().numpy()
    saliency2 = explainer(img2.cuda()).cpu().numpy()
    p1, c1 = torch.topk(model(img1.cuda()), k=top_k)  #torch.topk returns values and indices
    p1, c1 = p1[0], c1[0]
    p2, c2 = torch.topk(model(img2.cuda()), k=top_k)  #torch.topk returns values and indices
    p2, c2 = p2[0], c2[0]
    #Subplot 1
    plt.subplot(2, 2, 1)
    tensor_imshow(img1[0])
    #Subplot 2
    plt.subplot(2, 2, 2)
    tensor_imshow(img1[0])
    saliency1 = saliency1[c1[0]]
    plt.imshow(saliency1, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    #Subplot 3
    plt.subplot(2, 2, 3)
    tensor_imshow(img2[0])
    #Subplot 4
    plt.subplot(2, 2, 4)
    tensor_imshow(img2[0])
    saliency2 = saliency2[c2[0]]
    plt.imshow(saliency2, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.show()

    
# example_RISE(read_tensor('catdog.png'), 3)
# example_RISE(read_tensor('D:/E2ID/Research Papers/E2ID_Vis21/Code/RISE/Norm_Photos/faces/00002_931230_fa.jpg'),3)
example_corrRISE(read_tensor('D:/E2ID/Research Papers/E2ID_Vis21/Code/RISE/Norm_Photos/faces/00045_931230_fa_a.jpg'),read_tensor('D:/E2ID/Research Papers/E2ID_Vis21/Code/RISE/Norm_Photos/faces/00028_940128_fa.jpg'),1)

