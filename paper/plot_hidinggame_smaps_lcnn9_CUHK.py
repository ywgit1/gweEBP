# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:08:56 2022

@author: Yan Wang
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import random

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
    data_path = 'D:/Projects/E2ID/code/Composite2Photo/CUFS_dataset'
    matefilename = f'CUHK_test_{netname}_GT_as_Rank1.txt'
    nonmatefilename = f'CUHK_test_{netname}_Rank2_match_for_Rank1_GT.txt'
    smap_path = 'D:/Projects/E2ID/code/Composite2Photo/Visualization/hiding_game/hiding_game_results/saliency_maps'
    mate_df = pd.read_csv(os.path.join(data_path, matefilename), sep='@@@',  dtype=object, header=None, engine='python')
    nonmate_df = pd.read_csv(os.path.join(data_path, nonmatefilename), sep='@@@',  dtype=object, header=None, engine='python')
    probes = mate_df.iloc[::2, 0].tolist()
    probe_ids = mate_df.iloc[::2, 1].tolist()
    mates = mate_df.iloc[1::2, 0].tolist()
    nonmates = nonmate_df.iloc[1::2, 0].tolist()
    random.seed(0)
    selected_rows = random.sample(list(range(len(probes))), k=10)
    method_names = ['GradCAM', 'EBP', 'cEBP', 'tcEBP', 'PairwiseSIM', 'gweEBP']
    nr, nc = len(selected_rows), len(method_names) + 3   
    # # fig = plt.figure(figsize=(40*nc, 40*nr+40))
    fig, axes = plt.subplots(nr, nc, figsize=(15, 15))
    fig.tight_layout()
    
    for i, row in enumerate(selected_rows):
        probe_id = probe_ids[row]
        probe = probes[row]
        mate = mates[row]
        nonmate = nonmates[row]
        probe = mpimg.imread(probe)
        mate = mpimg.imread(mate)
        nonmate = mpimg.imread(nonmate)
        j = 0
        axes[i, j].imshow(probe)
        format_axes(axes[i, j])
        j += 1
        axes[i, j].imshow(mate)
        format_axes(axes[i, j])
        j += 1
        axes[i, j].imshow(nonmate)
        format_axes(axes[i, j])
        
        for method in method_names:
            if dataname == 'CUHK':
                if method == 'GradCAM':
                    smap = mpimg.imread(os.path.join(smap_path, \
                                                  f'{method}', f'{dataname}-{netname}', \
                                                f'{probe_id}-01-sz1_vis.png'))
                elif method == 'PairwiseSIM':
                    smap = mpimg.imread(os.path.join(smap_path, \
                                                  f'{method}', f'{dataname}-{netname}', \
                                                f'{probe_id}.jpg'))
                elif 'EBP' in method:
                    smap = mpimg.imread(os.path.join(smap_path, \
                                                  'EBP', f'{dataname}-{netname}', \
                                                f'{method}-{netname}-{dataname}-{probe_id}.jpg'))                    
            else:
                raise NotImplementedError()
            j += 1
            axes[i, j].imshow(smap)
            format_axes(axes[i, j])
      
    lbls = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
    for j in range(nc):
        axes[-1, j].set_xlabel(lbls[j], fontsize=30)
            
    plt.subplots_adjust(top=0.99, bottom=0.02,
                    wspace=0.01, 
                    hspace=0.01)
    # fig.tight_layout()
    
    plt.savefig(f"hiding_game-saliency_maps-{netname}-{dataname}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
        
