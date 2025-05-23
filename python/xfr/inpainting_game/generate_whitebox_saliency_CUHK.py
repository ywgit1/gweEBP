# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import os.path
import torch
import PIL
import numpy as np
import pdb
import uuid

import pandas as pd
import skimage
import skimage.morphology

import six
import itertools
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('agg')

import xfr
from xfr import utils
from xfr import inpaintgame3_dir
from xfr import inpaintgame_CUHK_saliencymaps_dir
from xfr.show import processSaliency
from xfr.show import create_save_smap

orig_image_pattern = '../{OriginalFile}'
inpainted_image_pattern = '../{InpaintingFile}'
mask_pattern = lambda dict_: \
    "../" + os.path.splitext(dict_['InpaintingFile'])[0] + "_mask" + \
    os.path.splitext(dict_['InpaintingFile'])[1]

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_subtree_triplet_ebp(wb, im_mates, im_nonmates, probe_im,
                             net_name,
                             ebp_version,
                             device,
                             ebp_percentile=50, topk=1,
                            ):
    """Subtree contrastive excitation backprop"""

    x_mates = []
    for im in im_mates:
        x_mates.append(
            wb.encode(wb.convert_from_numpy(im).to(device)).detach())
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmates.append(
            wb.encode(wb.convert_from_numpy(nm).to(device)).detach())
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)

    img_probe = wb.convert_from_numpy(probe_im).to(device)
    print("Calling Subtree EBP on img_probe")
    wb.net.set_triplet_classifier((1.0/2500.0)*avg_x_mate, (1.0/2500.0)*avg_x_nonmate)
    (img_subtree, P_subtree, k_subtree) = wb.subtree_ebp(
        img_probe,
        k_poschannel=0, k_negchannel=1,
        percentile=ebp_percentile, topk=topk,
    )
    print("Returning Subtree EBP")
    return img_subtree


def run_clrp(wb, im_mates, im_nonmates, probe_im, net_name, device):
    x_mates = []
    wb._clrp_mode = 'disable'    
    wb.net.restore_emd_layer(device=device) # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.encode(img_probe) # Set net.fc.indicator properly

    wb.net.set_triplet_classifier(x_probe, avg_x_mate, avg_x_nonmate)
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb.linear_forward_hook))
    hooks.append(wb.net.net.fc.register_backward_hook(wb.linear_backward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    # Add hooks to the triplet classification layer!
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb.linear_forward_hook))
    hooks.append(wb.net.net.fc2.register_backward_hook(wb.linear_backward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    img_saliency = wb.clrp(
        img_probe, target_class=0, device=device, contrastive_signal=True, CLRP='type1', k_R=8)    
    # 
    
    wb.layerlist.pop(-1)
    wb.layerlist.pop(-1)
    
    return img_saliency

def run_sess(wb, im_mates, im_nonmates, probe_im, net_name, method, device):
    x_mates = []   
    wb.net.restore_emd_layer(device=device) # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.encode(img_probe) # Set net.fc.indicator properly
    wb.net.set_triplet_classifier(x_probe, avg_x_mate, avg_x_nonmate)
    target_layer = None
    tokens = method.split('+')
    method = tokens[0]
    if 'LightCNN9' in str(wb.net):
        if method.lower() == 'gradcam':
            target_layer = 'features.29'
        else:
            target_layer = 'features.20'
    elif 'VGG16' in str(wb.net):
        if method.lower() == 'gradcam':
            target_layer = 'basenet_part2.13'
        else:
            target_layer = 'basenet_part2.9'
    img_saliency = wb.SESS(img_probe, tokens[0], target_layers=[target_layer])
    
    return img_saliency

def run_cam(wb, im_mates, im_nonmates, probe_im, net_name, method, device):
    x_mates = []   
    wb.net.restore_emd_layer(device=device) # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.encode(img_probe) # Set net.fc.indicator properly
    wb.net.set_triplet_classifier(x_probe, avg_x_mate, avg_x_nonmate)
    target_layer = None
    if 'LightCNN9' in str(wb.net):
        if method.lower() in ['gradcam', 'fdcam']:
            target_layer = wb.net.net.features[-2]
        else:
            target_layer = wb.net.net.features[-14] #-11
    elif 'VGG16' in str(wb.net):
        target_layer = wb.net.net.basenet_part2[-8]
    img_saliency = wb.CAM(img_probe, method, target_layers=[target_layer])
    
    return img_saliency

def run_agf(wb, im_mates, im_nonmates, probe_im, net_name, device):
    x_mates = []   
    wb.net.restore_emd_layer(device=device) # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.encode(img_probe) # Set net.fc.indicator properly
    wb.net.set_triplet_classifier(x_probe, avg_x_mate, avg_x_nonmate)
    i_layer = -4
    if 'LightCNN9' in str(wb.net):
        i_layer = -5
    elif 'VGG16' in str(wb.net):
        i_layer = -9
    img_saliency = wb.AGF(img_probe, i_layer=i_layer)
    
    return img_saliency    
   
def run_rsp(wb, im_mates, im_nonmates, probe_im, net_name, device):
    x_mates = []   
    wb.net.restore_emd_layer(device=device) # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.encode(img_probe) # Set net.fc.indicator properly
    wb.net.set_triplet_classifier(x_probe, avg_x_mate, avg_x_nonmate)
    i_layer = 0; gradcam_layer = 0
    if 'LightCNN9' in str(wb.net):
        i_layer = -12; gradcam_layer = 29 
    elif 'VGG16' in str(wb.net):
        i_layer = -6; gradcam_layer = 29
    else:
        raise ValueError(f'{str(wb.net)} net type not supported')
    img_saliency = wb.RSP(img_probe)
    
    return img_saliency 
 
def run_contrastive_triplet_ebp(wb, im_mates, im_nonmates, probe_im,
                                net_name,
                                ebp_version,
                                truncate_percent,
                                device,
                                # merge_layers=True
                       ):
    """ Contrastive excitation backprop"""
    # Add hooks to the embedding layer
 
    x_mates = []
    wb._ebp_mode = 'disable'    
    wb.net.restore_emd_layer() # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.encode(img_probe) # Set net.fc.indicator properly

    wb.net.set_triplet_classifier(x_probe, avg_x_mate,
                                  avg_x_nonmate)
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    # Add hooks to the triplet classification layer!
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc2.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    # Verify if the probe is classified as the mate

    wb._ebp_mode = 'disable'
    y = wb.net.classify(img_probe)
    y = y.detach().cpu().numpy()
    assert np.all(y[:, 0] > y[:, 1])
    
    k_mwp = -2
    if 'LightCNN9' in str(wb.net):
        k_mwp = 10
    elif 'VGG16' in str(wb.net):
        k_mwp = 8
        
    if truncate_percent is None:
        img_saliency = wb.contrastive_ebp(
            img_probe, k_poschannel=0, k_negchannel=1, k_mwp=k_mwp)
    else:
        img_saliency = wb.truncated_contrastive_ebp(
            img_probe, k_poschannel=0, k_negchannel=1,
            percentile=truncate_percent, k_mwp=k_mwp)

    wb.layerlist.pop(-1)
    wb.layerlist.pop(-1)
    # print("Returning Contrastive EBP")
    return img_saliency

def run_contrastive_triplet_eebp(wb, im_mates, im_nonmates, probe_im,
                                net_name,
                                ebp_version,
                                truncate_percent,
                                device,
                                # merge_layers=True
                       ):
    """ Contrastive excitation backprop"""

    x_mates = []
    wb.net.restore_emd_layer() # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)

    img_probe = wb.convert_from_numpy(probe_im).to(device)
    # print("Calling Contrastive EBP on img_probe")
    
    x_probe = wb.net.encode(img_probe) # Set net.fc.indicator properly
    
    wb.net.set_triplet_classifier(x_probe, avg_x_mate,
                                  avg_x_nonmate)
    # Add hooks to the embedding layer and triplet classification layer!
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc2.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    # Verify if the probe is classified as the mate
    y = wb.net.classify(img_probe)
    y = y.detach().cpu().numpy()
    assert np.all(y[:, 0] > y[:, 1])
    
    K = None
    k_mwp = -2
    if 'LightCNN9' in str(wb.net):
        K = 12
        k_mwp = 10
    elif 'VGG16' in str(wb.net):
        K = 9
        k_mwp = 8
    if truncate_percent is None:
        img_saliency = wb.contrastive_eebp(
            img_probe, k_poschannel=0, k_negchannel=1, K=K, k_mwp=k_mwp)
    else:
        img_saliency = wb.truncated_contrastive_eebp(
            img_probe, k_poschannel=0, k_negchannel=1,
            percentile=truncate_percent, K=K, k_mwp=k_mwp)

    wb.layerlist.pop(-1)
    wb.layerlist.pop(-1)
    # print("Returning Contrastive EBP")
    return img_saliency

def run_weighted_subtree_triplet_ebp(
    wb, im_mates, im_nonmates, probe_im,
    net_name,
    subtree_mode_weighted,
    ebp_version,
    device,
    # ebp_percentile=50,
    topk=1,
):
    """Subtree contrastive excitation backprop"""

    x_mates = []
    for im in im_mates:
        x_mates.append(wb.encode(wb.convert_from_numpy(im).to(device)).detach())
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmates.append(wb.encode(wb.convert_from_numpy(nm).to(device)).detach())
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)

    img_probe = wb.convert_from_numpy(probe_im).to(device)
    print("Calling weighted Subtree EBP on img_probe")
    wb.net.set_triplet_classifier(avg_x_mate, avg_x_nonmate)

    do_max_subtree=False
    do_mated_similarity_gating=False
    subtree_mode='norelu'
    """
    ebp_version= 7: Whitebox(...).weighted_subtree_ebp(...,
        do_max_subtree=True,
        subtree_mode='all',
        do_mated_similarity_gating=True)
    ebp_version= 8: Whitebox(...).weighted_subtree_ebp(...,
        do_max_subtree=False,
        subtree_mode='all',
        do_mated_similarity_gating=True)
    ebp_version= 9: Whitebox(...).weighted_subtree_ebp(...,
        do_max_subtree=True,
        subtree_mode='all',
        do_mated_similarity_gating=False)
    ebp_version=10: Whitebox(...).weighted_subtree_ebp(...,
        do_max_subtree=True,
        subtree_mode='norelu',
        do_mated_similarity_gating=True)
    ebp_version=11: Whitebox(..., with_bias=False).weighted_subtree_ebp (...,
        do_max_subtree=True,
        subtree_mode='all',
        do_mated_similarity_gating=True)
    """

    if ebp_version == 7:
        do_max_subtree = True
        # subtree_mode = 'all'
        do_mated_similarity_gating = True
    elif ebp_version == 8:
        do_max_subtree = False
        # subtree_mode = 'all'
        do_mated_similarity_gating = True
    elif ebp_version == 9:
        do_max_subtree = True
        # subtree_mode = 'all'
        do_mated_similarity_gating = False
    elif ebp_version == 10:
        do_max_subtree = True
        # subtree_mode = 'norelu'
        do_mated_similarity_gating = True
    elif ebp_version == 11:
        do_max_subtree = True
        # subtree_mode = 'all'
        do_mated_similarity_gating = True
    elif ebp_version == 12:
        do_max_subtree = False
        # subtree_mode = 'affineonly_with_prior'
        do_mated_similarity_gating = True

    (img_subtree, P_img, P_subtree, k_subtree) = wb.weighted_subtree_ebp(
        img_probe,
        k_poschannel=0, k_negchannel=1,
        topk=topk,
        do_max_subtree=do_max_subtree,
        subtree_mode=subtree_mode_weighted,
        do_mated_similarity_gating=do_mated_similarity_gating,
        )
    print("Returning weighted Subtree EBP")
    return img_subtree

def run_gradient_weighted_ecEBP(wb, im_mates, im_nonmates, probe_im,
                                net_name,
                                ebp_version,
                                device,
                                # merge_layers=True
                       ):
    """ Contrastive excitation backprop"""

    x_mates = []
    wb.net.restore_emd_layer()
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)

    img_probe = wb.convert_from_numpy(probe_im).to(device)
    # print("Calling Contrastive EBP on img_probe")

    x_probe = wb.net.encode(img_probe) # Set net.fc.indicator properly
    
    wb.net.set_triplet_classifier(x_probe, avg_x_mate,
                                  avg_x_nonmate)
    # Add hooks to the newly added triplet classification layer!
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc2.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    # Verify if the probe is classified as the mate
    y = wb.net.classify(img_probe)
    y = y.detach().cpu().numpy()
    assert np.all(y[:, 0] > y[:, 1])
    K = None
    k_mwp = -2
    if 'LightCNN9' in str(wb.net):
        K = 12
        k_mwp = 10
    elif 'VGG16' in str(wb.net):
        K = 9
        k_mwp = 8

    img_saliency = wb.gradient_weighted_ecEBP(
        img_probe, avg_x_mate, avg_x_nonmate, k_poschannel=0, k_negchannel=1, K=K, k_mwp=k_mwp)

    wb.layerlist.pop(-1)
    wb.layerlist.pop(-1)
    
    # print("Returning Contrastive EBP")
    return img_saliency

def run_gradient_weighted_eEBP(wb, im_mates, im_nonmates, probe_im,
                                net_name,
                                ebp_version,
                                device,
                                # merge_layers=True
                       ):
    """ Contrastive excitation backprop"""

    x_mates = []
    wb.net.restore_emd_layer()
    wb.layerlist.pop(-1)
    wb.layerlist.pop(-1)
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)

    img_probe = wb.convert_from_numpy(probe_im).to(device)
    # print("Calling Contrastive EBP on img_probe")

    x_probe = wb.net.encode(img_probe) # Set net.fc.indicator properly
    
    wb.net.set_triplet_classifier(x_probe, avg_x_mate,
                                  avg_x_nonmate)
    # Add hooks to the newly added triplet classification layer!
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc2.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    # Verify if the probe is classified as the mate
    y = wb.net.classify(img_probe)
    y = y.detach().cpu().numpy()
    assert np.all(y[:, 0] > y[:, 1])
    K = None
    k_mwp = -2
    if 'LightCNN9' in str(wb.net):
        K = 14
        k_mwp = 10 #10
    elif 'VGG16' in str(wb.net):
        K = 9
        k_mwp = 8

    img_saliency, A_list = wb.gradient_weighted_eEBP(
        img_probe, avg_x_mate, avg_x_nonmate, k_poschannel=0, k_negchannel=1, K=K, k_mwp=k_mwp)

    wb.layerlist.pop(-1)
    wb.layerlist.pop(-1)
    
    # print("Returning Contrastive EBP")
    return img_saliency

def run_negative_activation_analysis(wb, im_mates, im_nonmates, probe_im,
                                net_name,
                                ebp_version,
                                truncate_percent,
                                device,
                                # merge_layers=True
                       ):
    """ Contrastive excitation backprop"""
    # Add hooks to the embedding layer
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    
    if 'LightCNN' in str(wb.net):
        for i, layer in enumerate(wb.layerlist):
            if i < 23:
                for hook in layer['hooks']:
                    if hook is not None:
                        hook.remove()
        
    x_mates_ = []
    wb._ebp_mode = 'filter_negative_activation'    
    for im in im_mates:
        x_mate_ = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates_.append(x_mate_)
    x_nonmates_ = []
    for nm in im_nonmates:
        x_nonmate_ = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates_.append(x_nonmate_)
    avg_x_mate_ = torch.mean(torch.stack(x_mates_), axis=0)
    avg_x_mate_ /= torch.norm(avg_x_mate_)
    avg_x_nonmate_ = torch.mean(torch.stack(x_nonmates_), axis=0)
    avg_x_nonmate_ /= torch.norm(avg_x_nonmate_)

    img_probe = wb.convert_from_numpy(probe_im).to(device)
    # print("Calling Contrastive EBP on img_probe")
    x_probe_ = wb.encode(img_probe)   
    # avg_x_mate_ = torch.nn.functional.relu(avg_x_mate)
    # avg_x_nonmate_ = torch.nn.functional.relu(avg_x_nonmate)
    # x_probe_ = torch.nn.functional.relu(x_probe)
    # x_mate_ /= torch.norm(x_mate_)
    # x_nonmate_ /= torch.norm(x_nonmate_)
    # x_probe_ /= torch.norm(x_probe_)
    dp_1 = torch.sum(torch.mul(avg_x_mate_, x_probe_))
    dp_2 = torch.sum(torch.mul(avg_x_nonmate_, x_probe_))
    
    x_mates = []
    wb._ebp_mode = 'disable'    
    wb.net.restore_emd_layer() # Restore the original embedding layer
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device).requires_grad_(True)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
      
    x_probe = wb.encode(img_probe) # Set net.fc.indicator properly
    
    # if (dp_1 > dp_2):
    #     wb.tp += 1
    # else:
    #     wb.fn += 1

    wb.net.set_triplet_classifier(x_probe, avg_x_mate,
                                  avg_x_nonmate)
    # Add hooks to the triplet classification layer!
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc2.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    # Verify if the probe is classified as the mate

    wb._ebp_mode = 'disable'
    y = wb.net.classify(img_probe)
    y = y.detach().cpu().numpy()
    assert np.all(y[:, 0] > y[:, 1])

    return dp_1 > dp_2


def triplet_ebp(wb, im_mates, im_nonmates, probe_im, net_name, ebp_version, device):
    wb.net.restore_emd_layer() # Restore the original embedding layer
    
    x_mates = []    
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.net.encode(img_probe) # Set net.fc.indicator properly

    wb.net.set_triplet_classifier(x_probe, avg_x_mate, avg_x_nonmate)
    # Add hooks to the embedding layer and triplet classification layer!
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc2.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    # Verify if the probe is classified as the mate
    y = wb.net.classify(img_probe)
    y = y.detach().cpu().numpy()
    assert np.all(y[:, 0] > y[:, 1])
    
    
    x_probe = wb.convert_from_numpy(probe_im).to(device)
    k_mwp = -2
    if 'LightCNN9' in str(wb.net):
        k_mwp = 10
    elif 'VGG16' in str(wb.net):
        k_mwp = 8
    # Generate Excitation backprop (EBP) saliency map at first convolutional layer
    # P = torch.ones( (1, wb.net.num_classes()) )
    P = torch.zeros((1, 2))
    P[0][0] = 1.0
    P = P.to(device)
    wb._ebp_mode2 = 'pm' # probe is matched to mate
    img_saliency = wb.ebp(x_probe, P, k_mwp=k_mwp)
    return img_saliency


def triplet_eebp(wb, im_mates, im_nonmates, probe_im, net_name, ebp_version, device):
    wb.net.restore_emd_layer() # Restore the original embedding layer
    
    x_mates = []    
    for im in im_mates:
        x_mate = wb.encode(wb.convert_from_numpy(im).to(device)).detach()
        x_mates.append(x_mate)
    x_nonmates = []
    for nm in im_nonmates:
        x_nonmate = wb.encode(wb.convert_from_numpy(nm).to(device)).detach()
        x_nonmates.append(x_nonmate)
    avg_x_mate = torch.mean(torch.stack(x_mates), axis=0)
    avg_x_mate /= torch.norm(avg_x_mate)
    avg_x_nonmate = torch.mean(torch.stack(x_nonmates), axis=0)
    avg_x_nonmate /= torch.norm(avg_x_nonmate)
    
    img_probe = wb.convert_from_numpy(probe_im).to(device)
    x_probe = wb.net.encode(img_probe) # Set net.fc.indicator properly

    wb.net.set_triplet_classifier(x_probe, avg_x_mate, avg_x_nonmate)
    # Add hooks to the embedding layer and triplet classification layer!
    hooks = []    
    hooks.append(wb.net.net.fc.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc), 'hooks': hooks})
    hooks = []
    hooks.append(wb.net.net.fc2.register_forward_hook(wb._forward_hook))
    hooks.append(wb.net.net.fc2.register_forward_pre_hook(wb._preforward_hook))
    wb.layerlist.append({'name': str(wb.net.net.fc2), 'hooks': hooks})
    
    # Verify if the probe is classified as the mate
    y = wb.net.classify(img_probe)
    y = y.detach().cpu().numpy()
    assert np.all(y[:, 0] > y[:, 1])
    
    
    x_probe = wb.convert_from_numpy(probe_im).to(device)
    K = None
    k_mwp = -2
    if 'LightCNN9' in str(wb.net):
        K = 12
        k_mwp = 10
    elif 'VGG16' in str(wb.net):
        K = 9
        k_mwp = 8
    # Generate Excitation backprop (EBP) saliency map at first convolutional layer
    # P = torch.ones( (1, wb.net.num_classes()) )
    P = torch.zeros((1, 2))
    P[0][0] = 1.0
    P = P.to(device)
    img_saliency = wb.eebp(x_probe, P, K=K, k_mwp=k_mwp)
    return img_saliency

def shorten_subtree_mode(ebp_subtree_mode):
    if ebp_subtree_mode == 'affineonly_with_prior':
        return 'awp'
    return ebp_subtree_mode


def generate_wb_smaps(
    wb, net_name, img_base, subj_id,
    mask_id,
    subtree_mode_weighted,
    ebp_ver,
    overwrite,
    device,
    method,
    # merge_layers=True
):

    subject_id = subj_id

    multiprobe_data_dir = os.path.join(
        inpaintgame_CUHK_saliencymaps_dir,
        '{}-{}/{}'.format(
	method,
        net_name,
        subject_id))

    inpainting_v2_data = pd.read_csv(os.path.join(
        inpaintgame3_dir,
        'filtered_masks_threshold-{NET}.csv'.format(NET=net_name)))

    inpainting_v2_data = inpainting_v2_data.loc[
        (inpainting_v2_data['MASK_ID'] == int(mask_id)) &
        (inpainting_v2_data['SUBJECT_ID'] == subject_id)
    ]

    probe_data = []

    probes = [] # original probes
    mates = []  # original refs
    nonmates = []  # inpainted
    probe_masks = []
    probe_inpaints = []

    for idx, row in inpainting_v2_data.iterrows():
        d = row.to_dict()
        f = orig_image_pattern.format(**d)
        fm = mask_pattern(d)
        finp = inpainted_image_pattern.format(**d)

        if os.path.exists(f):
            if d['TRIPLET_SET'] == 'REF':
                mates.append(f)
            else:
                probe_data.append(row)
                probes.append(f)
                probe_masks.append(fm)
                probe_inpaints.append(finp)
        else:
            print('Original file %s does not exist!' % f)

        if d['TRIPLET_SET'] == 'REF':
            assert os.path.exists(finp)
            nonmates.append(finp)

    assert len(probes)==1
    im_mates = [im for im in utils.image_loader(mates)]
    im_nonmates = [im for im in utils.image_loader(nonmates)]

    probe_data = pd.DataFrame(probe_data)
 
    ntp = 0

    for probe_idx, (
        (probe_im, probe_fn), # image, filename
        probe_mask_fn,
        (_, probe_row)
    ) in enumerate(zip(
        [ret for ret in utils.image_loader(probes, returnFileName=True)],
        probe_masks,
        probe_data.iterrows()
    )):
        # Construct path if necessary
        # extra_dirs = os.path.split(
        #     os.path.relpath(probe_fn, cropped_data_dir))[0]
        output_dir = multiprobe_data_dir

        # print('\nOutput: %s\n' % output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # img_display = skimage.transform.resize(
        #     probe_im, (224, 224)
        # )
        # img_display = (img_display*255).astype(np.uint8)
        # img_probe = convert_from_numpy(probe_im).to(device)

        mask_im = [im for im in utils.image_loader([probe_mask_fn])][0]

        result_calculated = False

        if method is None or method=='EBP':
            result_calculated = True
            # fn = 'EBP_mode=%s_v%02d_%s' % (
            #     shorten_subtree_mode(wb.ebp_subtree_mode()),
            #     ebp_ver,
            #     device.type,
            # )
            fn = 'EBP'
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: triplet_ebp(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,                    
                    probe_im=probe_im,
                    net_name=net_name,
                    ebp_version=ebp_ver,
                    device=device,
                    # merge_layers=merge_layers
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )

        if method is None or method=='eEBP':
            result_calculated = True
            # fn = '%s_mode=%s_v%02d_%s' % (
            #     method,
            #     shorten_subtree_mode(wb.ebp_subtree_mode()),
            #     ebp_ver,
            #     device.type,
            # )
            fn = 'eEBP' # For Siva
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: triplet_eebp(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,                    
                    probe_im=probe_im,
                    net_name=net_name,
                    ebp_version=ebp_ver,
                    device=device,
                    # merge_layers=merge_layers
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )
            
        if method is None or method == 'cEBP':
            result_calculated = True
            for (truncate_percent) in [None, 20]:
                if truncate_percent is None:
                    # fn = 'contrastive_triplet_ebp_v%02d_%s' % (
                    # fn = '%s_mode=%s_v%02d_%s' % (
                    #     method,
                    #     shorten_subtree_mode(wb.ebp_subtree_mode()),
                    #     ebp_ver,
                    #     device.type,
                    # )
                    fn = 'cEBP'
                else:
                    # fn = 'trunc_contrastive_triplet_ebp_v%02d_pct%d_%s' % (
                    # fn = 't%s_mode=%s_v%02d_pct%d_%s' % (
                    #     method,
                    #     shorten_subtree_mode(wb.ebp_subtree_mode()),
                    #     ebp_ver,
                    #     truncate_percent,
                    #     device.type,
                    # )
                    fn = 'tcEBP'
                create_save_smap(
                    fn,
                    output_dir, overwrite,
                    smap_fn=lambda: run_contrastive_triplet_ebp(
                        wb=wb,
                        im_mates=im_mates,
                        im_nonmates=im_nonmates,
                        probe_im=probe_im,
                        truncate_percent=truncate_percent,
                        net_name=net_name,
                        ebp_version=ebp_ver,
                        device=device
                    ),
                    probe_im=probe_im,
                    probe_info=probe_row,
                    mask_im=mask_im,
                    mask_id=mask_id,
                )

        if method is None or method=='weighted-subtree':
            result_calculated = True
            for (topk) in [
                (32),
            ]:
                    # fn = 'weighted_subtree_triplet_ebp_v%02d_top%d_%s' % (
                    fn = 'weighted_subtree_triplet_ebp_mode=%s,%s_v%02d_top%d_%s' % (
                        shorten_subtree_mode(wb.ebp_subtree_mode()),
                        shorten_subtree_mode(subtree_mode_weighted),
                        ebp_ver,
                        topk,
                        # ebp_percentile,
                        device.type,
                    )
                    create_save_smap(
                        fn,
                        output_dir, overwrite,
                        smap_fn=lambda: run_weighted_subtree_triplet_ebp(
                            wb,
                            im_mates,
                            im_nonmates,
                            probe_im=probe_im,
                            # img_probe=img_probe,
                            # img_display=img_display,
                            # ebp_percentile=ebp_percentile,
                            topk=topk,
                            net_name=net_name,
                            ebp_version=ebp_ver,
                            subtree_mode_weighted=subtree_mode_weighted,
                            device=device,
                        ),
                        probe_im=probe_im,
                        probe_info=probe_row,
                        mask_im=mask_im,
                        mask_id=mask_id,
                    )
                    
        if method is None or method == 'ecEBP':
            result_calculated = True
            for (truncate_percent) in [20]:#, 20]:
                if truncate_percent is None:
                    # fn = 'contrastive_triplet_ebp_v%02d_%s' % (
                    # fn = '%s_mode=%s_v%02d_%s' % (
                    #     method,
                    #     shorten_subtree_mode(wb.ebp_subtree_mode()),
                    #     ebp_ver,
                    #     device.type,
                    # )
                    fn = 'ecEBP' # For Siva
                else:
                    # fn = 'trunc_contrastive_triplet_ebp_v%02d_pct%d_%s' % (
                    # fn = 'etcEBP_mode=%s_v%02d_pct%d_%s' % (
                    #     shorten_subtree_mode(wb.ebp_subtree_mode()),
                    #     ebp_ver,
                    #     truncate_percent,
                    #     device.type,
                    # )
                    fn = 'etcEBP' # For Siva
                create_save_smap(
                    fn,
                    output_dir, overwrite,
                    smap_fn=lambda: run_contrastive_triplet_eebp(
                        wb=wb,
                        im_mates=im_mates,
                        im_nonmates=im_nonmates,
                        probe_im=probe_im,
                        truncate_percent=truncate_percent,
                        net_name=net_name,
                        ebp_version=ebp_ver,
                        device=device
                    ),
                    probe_im=probe_im,
                    probe_info=probe_row,
                    mask_im=mask_im,
                    mask_id=mask_id,
                )

        if method is None or method.lower() == 'gwecebp':
            result_calculated = True
            fn = 'gwecEBP'
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: run_gradient_weighted_ecEBP(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,
                    probe_im=probe_im,
                    net_name=net_name,
                    ebp_version=ebp_ver,
                    device=device,
                    # merge_layers=merge_layers
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )

        if method is None or method.lower() == 'gweebp':
            result_calculated = True
            fn = 'gweEBP'
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: run_gradient_weighted_eEBP(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,
                    probe_im=probe_im,
                    net_name=net_name,
                    ebp_version=ebp_ver,
                    device=device,
                    # merge_layers=merge_layers
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )

        if method is None or method.lower() == 'clrp':
            result_calculated = True
            fn = 'cLRP'
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: run_clrp(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,
                    probe_im=probe_im,
                    net_name=net_name,
                    device=device
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )

        if 'sess' in method.lower():
            fn = method
            result_calculated = True
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: run_sess(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,
                    probe_im=probe_im,
                    net_name=net_name,
                    method=method,
                    device=device
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )             
        elif 'cam' in method.lower() or method.lower() == 'fullgrad':
            fn = method
            result_calculated = True
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: run_cam(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,
                    probe_im=probe_im,
                    net_name=net_name,
                    method=method,
                    device=device
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )            
            
        if method is None or method.lower() == 'agf':
            result_calculated = True
            fn = 'AGF'
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: run_agf(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,
                    probe_im=probe_im,
                    net_name=net_name,
                    device=device
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )
            
        if method is None or method.lower() == 'rsp':
            result_calculated = True
            fn = 'RSP'
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=lambda: run_rsp(
                    wb=wb,
                    im_mates=im_mates,
                    im_nonmates=im_nonmates,
                    probe_im=probe_im,
                    net_name=net_name,
                    device=device
                ),
                probe_im=probe_im,
                probe_info=probe_row,
                mask_im=mask_im,
                mask_id=mask_id,
            )
            
        if 'analysis' in method:
            result_calculated = True
            correct = run_negative_activation_analysis(
                        wb=wb,
                        im_mates=im_mates,
                        im_nonmates=im_nonmates,
                        probe_im=probe_im,
                        truncate_percent=None,
                        net_name=net_name,
                        ebp_version=ebp_ver,
                        device=device
                    )
            if correct:
                ntp += 1
                        
        if not result_calculated:
            raise RuntimeError(
                "Unknown method type %s (valid types: 'meanEBP', "
                "'contrastive', 'weighted-subtree')" % method
            )
            
    return ntp