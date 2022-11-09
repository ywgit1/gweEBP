# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:08:56 2022

@author: Yan Wang
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

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
    
    
    image_suffix = ['_OP', '_OS', '_FP', '_IP', '_FS', '_IS']
    ids = ['m-014', 'm-046']
    lbls = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    input_dir = 'data/cuhk'
    suffix2fn_all = []
    for id_ in ids:
        full_path = os.path.join(input_dir, id_)
        suffix2fn = {}
        files = os.listdir(full_path)
        for suffix in image_suffix:
            for fn in files:
                if suffix in fn:
                    suffix2fn[suffix] = os.path.join(full_path, fn)
                    break
        suffix2fn_all.append(suffix2fn)
    
    nr, nc = len(ids), len(image_suffix)   
    # fig = plt.figure(figsize=(40*nc, 40*nr+40))
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))    
    fig.tight_layout()
    i_plot = 1
    for i, s2f in enumerate(suffix2fn_all):
        for j, suffix in enumerate(image_suffix):
            img = mpimg.imread(s2f[suffix])
            # ax = fig.add_subplot(nr, nc, i_plot)
            # axes[i, j].set_xlim([0, 40])
            # axes[i, j].set_ylim([0, 40])
            axes[i, j].imshow(img)
            # plt.imshow(img)
            # ax.axis('equal')
            if i == nr - 1:
            # axes[i, j].set_title('xxx')
                axes[i, j].set_xlabel(lbls[j], fontsize=30)
            # plt.xlabel(lbls[j])
                # axes[i, j].axis('off')
            # i_plot += 1
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['bottom'].set_visible(False)
            axes[i, j].spines['left'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.subplots_adjust(top=0.99, bottom=0.02,
                    wspace=0.01, 
                    hspace=0.01)
    # fig.tight_layout()
    
    plt.savefig("cyclegan_cuhk_data.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
