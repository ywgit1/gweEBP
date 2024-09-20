# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:08:56 2022

@author: Yan Wang
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd

def format_axes(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xticks([])
    axis.set_yticks([])
    
if __name__ == "__main__":
    
    # fig, ax = plt.subplots(2, 2, figsize=(10,10))
    # fig.tight_layout()
    
    # #define data
    # x = [1, 2, 3]
    # y = [7, 13, 24]
    
    #create subplots
    # ax[0, 0].plot(x, y, color='red')
    # ax[0, 1].plot(x, y, color='blue')
    # ax[1, 0].plot(x, y, color='green')
    # ax[1, 1].plot(x, y, color='purple')
    # plt.show()
    dataname = 'CUHK'
    netname = 'lcnn9'
    datafilename = f'../data/inpainting-game/{dataname}/filtered_masks_threshold-{dataname}-{netname}.csv'
    smap_path = f'../data/inpainting-game-saliency-maps/{dataname}'
    df = pd.read_csv(datafilename, sep=',',  header=None, engine='python')
    # subject_ids = df.iloc[:, 0]
    # mask_ids = df.iloc[:, 1]
    # original_files = df.iloc[:, 2]
    # inpainted_files = df.iloc[:, 3]
    selected_ids = ['m2-034', 'f1-013', 'm2-096', 'f1-013', 'm1-020', 'f1-006', 'm2-097', 'm2-039', 'm2-061', 'm2-034']
    selected_masks = ['0', '3', '1', '2', '3', '4', '0', '1', '4', '2']
    method_names = ['GradCAM', 'EBP', 'cEBP', 'tcEBP', 'PairwiseSIM', 'XFace', 'CorrRISE', 'gweEBP']
    method_labels = ['GradCAM', 'EBP', 'cEBP', 'tcEBP', 'PairwiseSIM', 'bbox-xface', 'bbox-corrrise_perct=10_scale_12', 'gweEBP']
    nr, nc = len(selected_ids), len(method_names) + 4   
    # fig = plt.figure(figsize=(40*nc, 40*nr+40))
    fig, axes = plt.subplots(nr, nc, figsize=(15, 15))
    fig.tight_layout()
    
    
    for i, (sid, smask) in enumerate(zip(selected_ids, selected_masks)):
        probe_row = df[(df.iloc[:, 0] == sid) & (df.iloc[:, 1] == smask) & (df.iloc[:, -1] == 'PROBE')]
        ref_row = df[(df.iloc[:, 0] == sid) & (df.iloc[:, 1] == smask) & (df.iloc[:, -1] == 'REF')]
        probe = mpimg.imread(probe_row.iloc[0, 4])
        mate = mpimg.imread(ref_row.iloc[0, 4])
        inpainted_mate = mpimg.imread(ref_row.iloc[0, 5])
        mask = mpimg.imread(ref_row.iloc[0, 3] + '_mask.png')
        j = 0
        axes[i, j].imshow(probe)
        format_axes(axes[i, j])
        j += 1
        axes[i, j].imshow(mate)
        format_axes(axes[i, j])
        j += 1
        axes[i, j].imshow(mask)
        format_axes(axes[i, j])
        j += 1
        axes[i, j].imshow(inpainted_mate)
        format_axes(axes[i, j])
        
        for method, label in zip(method_names, method_labels):
            if dataname == 'CUHK':
                smap = mpimg.imread(os.path.join(smap_path, \
                                                 f'{method}-{netname}', \
                                                f'{sid}', \
                                                f'{smask}-{label}-saliency-overlay.png'))
            else:
                raise NotImplementedError()
            j += 1
            axes[i, j].imshow(smap)
            format_axes(axes[i, j])
      
    lbls = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)']
    for j in range(nc):
        axes[-1, j].set_xlabel(lbls[j], fontsize=30)
            
    plt.subplots_adjust(top=0.99, bottom=0.02,
                    wspace=0.01, 
                    hspace=0.01)
    # fig.tight_layout()
    
    plt.savefig(f"inpainting_game-saliency_maps-{netname}-{dataname}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
        
    
