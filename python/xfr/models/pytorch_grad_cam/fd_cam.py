# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:21:41 2023

@author: User1
Li et al. (2022) "FD-CAM: Improving Faithfulness and Discriminability of Visual Explanation for CNNs"
"""

from itertools import combinations
import cv2
import numpy as np
import torch
import ttach as tta
import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from xfr.models.pytorch_grad_cam.base_cam import BaseCAM
from xfr.models.pytorch_grad_cam.utils.model_targets import EmbeddingNetTripletGainTarget




class FDCAM(BaseCAM):
    def __init__(self,
                 model, 
                 target_layers, 
                 threshold=0.95, 
                 use_cuda=False,
                 reshape_transform=None):

        super(FDCAM, self).__init__(model, target_layers, use_cuda=use_cuda, 
            reshape_transform=reshape_transform)
        self.model = model
        self.threshold = threshold
        self.target_layers = target_layers

    def minMax(self,tensor):
        maxs = tensor.max(dim=1)[0]
        mins = tensor.min(dim=1)[0]
        return (tensor-mins)/(maxs-mins)

    def scaled(self,tensor):
        maxs = tensor.max(dim=1)[0]
        return (tensor)/(maxs)

    def get_cos_similar_matrix(self,v1, v2):
        num = np.dot(v1, np.array(v2).T) 
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1) 
        res = num / denom
        res[np.isnan(res)] = 0
        return res

    
    def combination(self,scores,grads_tensor):

        grads = self.minMax(grads_tensor)
        scores = self.minMax(scores)

        weights = torch.exp(scores) * grads - 0.5

        
        return weights

        

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        with torch.no_grad():             
            grads = np.mean(grads, axis=(2, 3)) 
            BATCH_SIZE = 1
            
            activation = activations.reshape(activations.shape[1],-1)
            cosine = self.get_cos_similar_matrix(activation, activation)
            activation_tensor = torch.from_numpy(activations)
            
            cosine = torch.from_numpy(cosine)
            activation = torch.from_numpy(activation)

            record0 = torch.ones(cosine.shape).cuda()
            record1 = torch.zeros(cosine.shape).cuda()

            
            for i in range(cosine.shape[0]):
                threshold0 = torch.quantile(cosine[i,:],self.threshold)
                record1[i,:] = cosine[i,:]>threshold0

            record2 = record0-record1

            if self.cuda:
                activation_tensor = activation_tensor.cuda()
                grad_tensor = torch.from_numpy(grads).cuda()
                record1 = record1.cuda()
                record2 = record2.cuda()

            scores = []
            # orig_result = np.float32(self.model(input_tensor)[:,target_category].cpu()).reshape(1,)
            number_of_channels = activation_tensor.shape[1]
            for tensor, category in zip(activation_tensor, target_category):
                orig_result = np.float32(category(self.model(input_tensor)).cpu()).reshape(1,)
                batch_tensor = tensor.repeat(BATCH_SIZE, 1, 1, 1)
                for i in range(0, number_of_channels, BATCH_SIZE):
                    batch = batch_tensor * record1[i:i + BATCH_SIZE,:,None,None]     ## on
                    # score = self.model.classifier(torch.flatten(self.model.avgpool(batch),1))[:, category].cpu().numpy()
                    outputs = self.model.classifier(torch.flatten(self.model.avgpool(batch), 1))
                    # outputs = self.model.classify(batch)
                    score = [category(o).cpu().numpy() for o in outputs]          
                    batch = batch_tensor*record2[i:i + BATCH_SIZE,:,None,None]     ## off
                    # score += orig_result - self.model.classifier(torch.flatten(self.model.avgpool(batch),1))[:, category].cpu().numpy().reshape(BATCH_SIZE,)
                    outputs = self.model.classifier(torch.flatten(self.model.avgpool(batch), 1))
                    # outputs = self.model.classify(batch)
                    score = [s + orig_result - category(o).cpu().numpy() for s, o in zip(score, outputs)]
                    scores.extend(score)
  

            scores = np.float32(scores).reshape(activations.shape[0], activations.shape[1])
            # scores = scores/(2*orig_result)              ## off+on relative 
 
            scores = torch.tensor(scores).cuda()
            scores = self.combination(scores,grad_tensor).cpu().numpy()
            
            return scores

