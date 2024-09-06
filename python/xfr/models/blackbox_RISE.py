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
# from xfr.models.RISE.explanations import RISE 
from tqdm import tqdm


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


class BlackBoxRISE:
    def __init__(self,
            probe=None,
            refs=None,
            ref_sids = None,
            potential_gallery=None,
            gallery=None, 
            gallery_size=50,
            black_box_fn=None,
            num_masks=5000,
            input_size=224,
            mask_scale=12,
            perct = 10,
            triplet_score_type='cts',
            use_gpu=True,
            device=None,
        ):
        
        self.triplet_scoring_fns = {'cts': self.contrastive_triplet_similarity}

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
        self.black_box_fn = black_box_fn

        self.num_masks = num_masks
        self.mask_scale = mask_scale

        # Setup triplet scoring function
        if (triplet_score_type is not None):
            if (triplet_score_type not in self.triplet_scoring_fns):
                raise ValueError('Specified triplet score type "{}" is not supported.'.format(triplet_score_type))
            else:
                self.triplet_score_type = triplet_score_type
                self.triplet_scoring_fn = self.triplet_scoring_fns[triplet_score_type]
        else:
            raise ValueError('Triplet score type must be specified')
            
        self.perct = perct
        self.input_size = input_size
        self.apply_masks = self.mask_fill_gray
    
    def set_probe(self, probe):
        if isinstance(probe, str) or isinstance(probe, np.ndarray):
            self.probe = center_crop(probe, convert_uint8=False)
        else:
            raise ValueError('Probe must be a filepath to an image or a NumPy array')

        # Reset probe gallery scores if necessary
        if (hasattr(self, 'original_probe_gallery_scores')):
            self.original_probe_gallery_scores = None 

    def apply_masks_using_image(self, image):
        masked_images = np.zeros(((self.num_masks,) + image.shape), dtype=np.float32)

        # Blend between probe and image according to masks
        for i, mask in enumerate(self.masks):
            masked_image = mask[..., np.newaxis] * self.probe + (1.0-mask[...,np.newaxis]) * image
            masked_images[i,...] = masked_image
        self.masked_probes = masked_images

    def mask_fill_gray(self):
        fill_image = 0.5 * np.ones(self.probe.shape)
        self.apply_masks_using_image(fill_image)

    def mask_fill_blur(self):
        blurred = skimage.filters.gaussian(
            self.probe,
            self.blur_fill_sigma_percent / 100.0 * max(self.probe.shape),
            multichannel=True,
            preserve_range=True
        )
        self.apply_masks_using_image(blurred)
        
    def generate_sparse_masks(self, random_shift=True, order=1):
        s = self.mask_scale
        N = self.num_masks
        p1 = self.perct * 0.01
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) > p1 # p1 is small, so only small patches are occluded
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size), dtype=np.float32)

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        # self.masks = self.masks.reshape(-1, *self.input_size)

    def contrastive_triplet_similarity(self):
        ref_scores = self.original_probe_ref_scores - self.masked_probe_ref_scores
        gallery_scores = self.original_probe_gallery_scores - self.masked_probe_gallery_scores
        scores = (ref_scores - gallery_scores).mean(axis=1)
        # scores = (self.masked_probe_ref_scores - self.masked_probe_gallery_scores).mean(axis=1)
        return scores

    def score_masks(self):
        # Compute original ref black box scores
        self.original_probe_ref_scores = self.black_box_fn([self.probe], self.refs) 

        # Compute original gallery black box scores
        if (not hasattr(self, 'original_probe_gallery_scores') or
                self.original_probe_gallery_scores is None):
            self.original_probe_gallery_scores = self.black_box_fn([self.probe], self.gallery)

        # Compute perturbed ref black box scores
        self.masked_probe_ref_scores = self.black_box_fn(
            self.masked_probes, self.refs)

        # Compute perturbed gallery black box scores
        self.masked_probe_gallery_scores = self.black_box_fn(
            self.masked_probes, self.gallery)

        # Compute mask scores using a triplet function
        self.mask_scores = self.triplet_scoring_fn()

    def combine_masks(self, indices=None):
        if indices is None:
            filtered_weights = self.mask_scores[:]
            filtered_masks = self.masks
        else:
            filtered_weights = self.mask_scores[indices]
            filtered_masks = self.masks[indices,...]

        weighted_masks = filtered_weights[...,np.newaxis,np.newaxis] * filtered_masks
        combination = weighted_masks.mean(axis=0)
        
        return combination

    def compute_saliency_map(self, positive_scores=True, percentile=0):
        # Sort mask scores
        sorted_idx = self.mask_scores.argsort()[::-1]
        pos_sorted_idx = sorted_idx[self.mask_scores[sorted_idx] > 0]
        neg_sorted_idx = sorted_idx[self.mask_scores[sorted_idx] < 0][::-1]

        # try: - most of the time it indicates wrong pair of images used
        # Select indices based on percentile
        if (positive_scores):
            threshold = np.percentile(self.mask_scores[pos_sorted_idx], percentile)
            selected_indices = self.mask_scores >= threshold
            saliency_map = 1.0-self.combine_masks(selected_indices)
        else:
            threshold = np.percentile(-self.mask_scores[neg_sorted_idx], percentile)
            selected_indices = -self.mask_scores >= threshold
            saliency_map = self.combine_masks(selected_indices)-1.0
        
        # saliency_map = 1.0 - self.combine_masks()

        saliency_map -= saliency_map.min()
        saliency_map /= saliency_map.max()
        self.saliency_map = saliency_map

    def evaluate(self):
        curr_step = 1
        num_steps = 4

        #if (self.gallery is None):
        #    num_steps += 1
        #    print_flush('{}/{} Building gallery...'.format(curr_step, num_steps), flush=True)
        #    self.build_gallery()
        #    curr_step += 1
        
        print_flush('\n{}/{} Generating masks...'.format(curr_step, num_steps), flush=True)
        self.generate_sparse_masks()
        curr_step += 1
        
        print_flush('\n{}/{} Applying masks...'.format(curr_step, num_steps), flush=True)
        self.apply_masks()
        curr_step += 1
        
        print_flush('\n{}/{} Scoring masks...'.format(curr_step, num_steps), flush=True)
        self.score_masks()
        curr_step += 1
        
        print_flush('\n{}/{} Computing saliency map...'.format(curr_step, num_steps), flush=True)
        self.compute_saliency_map()
        
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
