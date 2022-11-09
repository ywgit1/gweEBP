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
    dataname = 'UoM'
    netname = 'lcnn9'
    datafilename = f'../data/inpainting-game/{dataname}/filtered_masks_threshold-{netname}.csv'
    smap_path = f'../data/inpainting-game-saliency-maps/{dataname}'
    df = pd.read_csv(datafilename, sep=',',  header=None, engine='python')
    subject_ids = df.iloc[:, 0]
    mask_ids = df.iloc[:, 1]
    original_files = df.iloc[:, 2]
    inpainted_files = df.iloc[:, 3]
    selected_ids = ['2', '46', '49', '56', '59', '75', '83', '89', '101', '193']
    selected_masks = ['0', '3', '3', '4', '2', '1', '4', '0', '2', '1']
    method_names = ['GradCAM', 'EBP', 'cEBP', 'tcEBP', 'PairwiseSIM', 'gweEBP']
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
        
        for method in method_names:
            if dataname == 'UoM':
                smap = mpimg.imread(os.path.join(smap_path, \
                                                 f'{method}-{netname}', \
                                                '{:05d}'.format(int(sid)), \
                                                f'{smask}-{method}-saliency-overlay.png'))
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
    
    plt.savefig(f"inpainting_game-saliency_maps-{netname}-{dataname}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
        
    # lbls = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    # input_dir = 'data/cuhk'
    # suffix2fn_all = []
    # for id_ in ids:
    #     full_path = os.path.join(input_dir, id_)
    #     suffix2fn = {}
    #     files = os.listdir(full_path)
    #     for suffix in image_suffix:
    #         for fn in files:
    #             if suffix in fn:
    #                 suffix2fn[suffix] = os.path.join(full_path, fn)
    #                 break
    #     suffix2fn_all.append(suffix2fn)
    


    # for i, s2f in enumerate(suffix2fn_all):
    #     for j, suffix in enumerate(image_suffix):
    #         img = mpimg.imread(s2f[suffix])
    #         # ax = fig.add_subplot(nr, nc, i_plot)
    #         # axes[i, j].set_xlim([0, 40])
    #         # axes[i, j].set_ylim([0, 40])
    #         axes[i, j].imshow(img)
    #         # plt.imshow(img)
    #         # ax.axis('equal')
    #         if i == nr - 1:
    #         # axes[i, j].set_title('xxx')
    #             axes[i, j].set_xlabel(lbls[j], fontsize=30)
    #         # plt.xlabel(lbls[j])
    #             # axes[i, j].axis('off')
    #         # i_plot += 1
    #         axes[i, j].spines['top'].set_visible(False)
    #         axes[i, j].spines['bottom'].set_visible(False)
    #         axes[i, j].spines['left'].set_visible(False)
    #         axes[i, j].spines['right'].set_visible(False)
    #         axes[i, j].set_xticklabels([])
    #         axes[i, j].set_yticklabels([])
    #         axes[i, j].set_xticks([])
    #         axes[i, j].set_yticks([])
    # plt.subplots_adjust(top=0.99, bottom=0.02,
    #                 wspace=0.01, 
    #                 hspace=0.01)
    # # fig.tight_layout()
    
    # plt.savefig("cyclegan_cuhk_data.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    
