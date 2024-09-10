# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import math
import shutil
import subprocess
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

import skimage
from skimage.transform import resize
import skimage.filters

import torch
from xfr.utils import create_net
from xfr.utils import center_crop
from xfr.models.resnet import convert_resnet101v4_image
from xfr.models.XFace.xface import XFace, normalize
from tqdm import tqdm
import cv2
from xfr import utils
import torch

def print_flush(str, file=sys.stdout, flush=True):
    file.write(str + '\n')
    if flush:
        file.flush()

have_gpu_state = False
can_use_gpu = False
def check_gpu(use_gpu=True):
    # taa: This is pretty massively incorrect.
    # nvidia-smi does not honor CUDA_VISIBLE_DEVICES (it's not a CUDA app,
    # why should it?).
    # Add call to cuda.is_available, which at least covers the case
    # where CUDA_VISIBLE_DEVICES is empty.
    global have_gpu_state
    global can_use_gpu
    if not torch.cuda.is_available():
        have_gpu_state = True
        can_use_gpu = False
        return False
    if have_gpu_state:
        return can_use_gpu
    have_gpu_state = True
    if not use_gpu:
        can_use_gpu = False
    elif 'STR_USE_GPU' in os.environ and len(os.environ['STR_USE_GPU']) > 0:
        try:
            gpu = int(os.environ['STR_USE_GPU'])
            if gpu >= 0:
                can_use_gpu = True
            else:
                can_use_gpu = False
        except ValueError as e:
            can_use_gpu = False
    else:
        import subprocess
        # Ask nvidia-smi for list of available GPUs
        try:
            proc = subprocess.Popen(["nvidia-smi", "--list-gpus"],
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            gpu_list = []
            if proc:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    linelist = line.split()
                    if linelist[0] == b'GPU':
                        gpu_id = linelist[1].decode("utf-8")[0]
                        gpu_list.append(gpu_id)
            if len(gpu_list) > 0:
                can_use_gpu = True
        except:
            can_use_gpu = False
    return can_use_gpu


def custom_black_box_fn(probes, gallery):
    """
    User defined black box scoring function.

    This function should compute pairwise similarity scores between all images
    in the 'probes' and 'gallery'. Within this function you should pass image
    data to your black box system and read back in the scores it returns. Note
    the data type requirements for the args and returns below.
    
    To use your custom black box function, instantiate the STRise object by
    setting the 'black_box_fn' parameter to be the name of your custom black box
    function, i.e. STRise(black_box_fn=custom_black_box_fn). A specific example
    illustrating this for the PittPatt system can be found in the following 
    Jupyter notebook: xfr/demo/blackbox_demo_pittpatt.ipynb

    Args:
        probes: A list of numpy arrays of images
        gallery: A list of filepaths to or numpy arrays of images

    Returns:
        A numpy array of size [len(probes) len(gallery)]. Contains
        the similarity score between the ith probe and the jth gallery
        image in the ith row and jth column.
    """
    pass


class BlackBoxXFace:
    def __init__(self,
            probe=None,
            refs=None,
            ref_sids = None,
            potential_gallery=None,
            gallery=None, 
            gallery_size=50,
            black_box_fn=None, # pass the model instead of function
            use_gpu=True,
            device=None,
        ):

        # Validate inputs
        self.use_gpu = check_gpu(use_gpu=use_gpu)
        if not self.use_gpu:
            self.device = torch.device("cpu")
        else:
            if device is None:
                warnings.warn('use_gpu of STRise will be deprecated, use device',
                              PendingDeprecationWarning)
                self.device = torch.device('cuda')
            else:
                self.use_gpu = None
                self.device = device
        
        # Setup probe and reference
        if (probe is not None and refs is not None):
            if isinstance(probe, str) or isinstance(probe, np.ndarray):
                self.probe = center_crop(probe, convert_uint8=True)
            else:
                raise ValueError('Probe must be a filepath to an image or a NumPy array')

            if isinstance(refs, list) or isinstance(refs, np.ndarray) or isinstance(refs, pd.DataFrame):
                self.refs = refs 
            else:
                raise ValueError('Refs must be a list of filepaths, NumPy arrays, or a Pandas dataframe') 
            self.ref_sids = ref_sids
        else:
            raise ValueError('Probe and reference must be specified')

        # Setup potential gallery
        if (potential_gallery is not None):
            self.potential_gallery = potential_gallery
            if isinstance(potential_gallery, list):
                # List
                self.potential_gallery_size = len(potential_gallery)
            elif isinstance(potential_gallery, np.ndarray):
                # NumPy array
                self.potential_gallery_size = potential_gallery.shape[0]
            elif isinstance(potential_gallery, pd.DataFrame):
                # Pandas dataframe
                self.potential_gallery_size = len(potential_gallery.index)
            else:
                raise TypeError('Potential gallery must be a list of filepaths, NumPy arrays, or a Pandas dataframe')
        else:
            self.potential_gallery = potential_gallery

        # Setup gallery
        if (gallery is not None):
            self.gallery = gallery
            if isinstance(gallery, list):
                # List
                self.gallery_size = len(gallery)
            elif isinstance(gallery, np.ndarray):
                # NumPy array
                self.gallery_size = gallery.shape[0]
            elif isinstance(gallery, pd.DataFrame):
                # Pandas dataframe
                self.gallery_size = len(gallery.index)
            else:
                raise TypeError('Gallery must be a list of filepaths, NumPy arrays, or a Pandas dataframe')
        else:
            self.gallery = gallery
            self.gallery_size = gallery_size

        # Setup black box
        def inference_func(img, model):
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3:
                    img = model.convert_from_numpy(img) # ndarray->torch.tensor
                    emd = model.embeddings(img)
                    return emd.reshape(-1)
                elif len(img.shape) == 4:
                    imgs = []
                    for im in tqdm(img):
                        im = model.convert_from_numpy(im)
                        imgs.append(im)
                    imgTensor = torch.cat(imgs, dim=0)
                    emd_list = []
                    for i in tqdm(range(0, imgTensor.shape[0], 64)):
                        batch = imgTensor[i:min(i + 64, imgTensor.shape[0])]
                        emd_list.append(model.embeddings(batch))
                    emd = np.concatenate(emd_list, axis=0)
                    return emd
            elif isinstance(img, str):
                img = [im for im in utils.image_loader([img])][0]
                img = model.convert_from_numpy(img)
                emd = model.embeddings(img)
                return emd
            else:
                raise TypeError('Bad image type {}'.format(type(img)))
            
        self.xFace = XFace(black_box_fn, inference_func)

    def evaluate(self):
        ref = self.refs[0]
        if isinstance(ref, str):
            ref = [im for im in utils.image_loader([ref])][0]
        sal_pp = self.xFace(self.probe, ref)
        gallery = self.gallery[0]
        if isinstance(gallery, str):
            gallery = [im for im in utils.image_loader([gallery])][0]
        sal_np = self.xFace(self.probe, gallery)
        saliency_map = sal_pp - sal_np
        self.saliency_map = normalize(saliency_map)
        print_flush('\nFinished!')

    def plot_gallery(self):
        ncols = 10
        nrows = int(math.ceil(1.0 * self.gallery_size / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols,nrows))
        if isinstance(self.gallery, pd.DataFrame):
            for i, id in enumerate(self.gallery.index):
                im = center_crop(self.gallery.at[id,'Filename'], convert_uint8=False)
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)
        else:
            for i, im in enumerate(self.gallery):
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)

        for ii in range(i+1, nrows*ncols):
            fig.delaxes(axes.flat[ii])

        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.show()

    def save_gallery(self, filename):
        ncols = 10
        nrows = int(math.ceil(1.0 * self.gallery_size / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols,nrows))
        if isinstance(self.gallery, pd.DataFrame):
            for i, id in enumerate(self.gallery.index):
                im = center_crop(self.gallery.at[id,'Filename'], convert_uint8=False)
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)
        else:
            for i, im in enumerate(self.gallery):
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)

        for ii in range(i+1, nrows*ncols):
            fig.delaxes(axes.flat[ii])

        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(filename, bbox_inches='tight') 
