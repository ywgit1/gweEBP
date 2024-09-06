import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s) #array([28., 28.])
        up_size = (s + 1) * cell_size  #array([252., 252.])

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')                #(6000, 8, 8)

        self.masks = np.empty((N, *self.input_size))  #(6000, 224, 224)

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])  
            y = np.random.randint(0, cell_size[1])  
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size) #(6000, 1, 224, 224)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = 0.1
    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)   #torch.Size([6000, 3, 224, 224])
        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
            
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)  #1000 (out_features = 1000 from resnet50 fully connected layer)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W)) #torch.size([1000,6000]) matmul torch.Size([6000, 50176])
        sal = sal.view((CL, H, W))  #torch.Size([1000, 224, 224])
        sal = sal / N / self.p1   # very small values between 0 and 1
        return sal
    
# Modified by Siva
class corrRISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(corrRISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s) #array([28., 28.])
        up_size = (s + 1) * cell_size  #array([252., 252.])

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')                #(6000, 8, 8)

        self.masks = np.empty((N, *self.input_size))  #(6000, 224, 224)

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])  
            y = np.random.randint(0, cell_size[1])  
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size) #(6000, 1, 224, 224)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = 0.1
    def forward(self, img1 , img2):
        N = self.N
        _, _, H1, W1 = img1.size()
        _, _, H2, W2 = img2.size()
        
        # Check if the sizes are the same using assert
        assert(H1,W1) == (H2,W2), f"AssertionError: The two images have different sizes: Image 1 is ({H1}, {W1}), Image 2 is ({H2}, {W2})"
        print("The two images have the same size.")
        
        """ Step 1: Calculating image embeddings"""
        x=self.model(img1.cuda()) #image1 embedding
        y=self.model(img2.cuda()) #image2 embedding
        
        """ Step 2: Calculating Masked image embeddings """
        # Apply array of filters to the image
        stack1 = torch.mul(self.masks, img1.data)   #torch.Size([6000, 3, 224, 224])
        stack2 = torch.mul(self.masks, img2.data)   #torch.Size([6000, 3, 224, 224])
        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        
        x_m = []
        y_m = []
        for i in range(0, N, self.gpu_batch):
            x_m.append(self.model(stack1[i:min(i + self.gpu_batch, N)]))
            y_m.append(self.model(stack2[i:min(i + self.gpu_batch, N)]))
            
        x_m = torch.cat(x_m)  #torch.Size([6000, 2048, 1, 1]) # masked image1 embedding
        y_m = torch.cat(y_m)  #torch.Size([6000, 2048, 1, 1])  # masked image2 embedding
        
        """ Step 3: Calculating Cosine similarity between the masked image embeddings of one image against the image embedding of another image """
        #Flatten the 4Dtensor into 2D tensor and convert into numpy
        x=x.reshape(x.size(0),-1).cpu().numpy()
        y=y.reshape(y.size(0),-1).cpu().numpy()
        
        x_m=x_m.reshape(x_m.size(0),-1).cpu().numpy() #Convert tensor into numpy 
        y_m=y_m.reshape(y_m.size(0),-1).cpu().numpy() 
        
        sc_x=cosine_similarity(x_m,y) # (6000, 1) similarity scores
        sc_y=cosine_similarity(y_m,x) # (6000, 1) similarity scores
        
        """ Step 4: Calculating Pearsons correlation coefficient between the cosine sim score and all the masks at a specific location """
        # Initialize 
        pearson_corr_x = np.zeros((H1, W1))
        pearson_corr_y = np.zeros((H1, W1))

        for idx in range(H1):
            for jdx in range(W1):
                pearson_corr_x[idx,jdx], _ = pearsonr(sc_x.reshape(-1), self.masks[:,:,idx,jdx].reshape(-1).cpu().detach().numpy()) # returns pearsons correlation and it's p value
                pearson_corr_y[idx,jdx], _ = pearsonr(sc_y.reshape(-1), self.masks[:,:,idx,jdx].reshape(-1).cpu().detach().numpy())
                # print(f"Pearson's Correlation (SciPy): {pearson_corr:.4f}")
        sal_x=pearson_corr_x
        sal_y=pearson_corr_y
        
        return [sal_x, sal_y]

class corrRISEBatch(corrRISE):
    def forward(self, img1 , img2):
        N = self.N
        B, C, H1, W1 = img1.size()
        B, C, H2, W2 = img2.size()
        
        # Check if the sizes are the same using assert
        assert(H1,W1) == (H2,W2), f"AssertionError: The two images have different sizes: Image 1 is ({H1}, {W1}), Image 2 is ({H2}, {W2})"
        print("The two images have the same size.")
        
        """ Step 1: Calculating image embeddings"""
        x=self.model(img1.cuda()) #image1 embedding
        y=self.model(img2.cuda()) #image2 embedding
        
        """ Step 2: Calculating Masked image embeddings """
        # Apply array of filters to the image
        stack1 = torch.mul(self.masks.view(N,1,H1,W1), img1.data.view(B*C,H1,W1))   #torch.Size([6000, 3, 224, 224])
        stack2 = torch.mul(self.masks.view(N,1,H2,W2), img2.data.view(B*C,H2,W2))   #torch.Size([6000, 3, 224, 224])
        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        
        x_m = []
        y_m = []
        for i in range(0, N*B, self.gpu_batch):
            x_m.append(self.model(stack1[i:min(i + self.gpu_batch, N*B)]))
            y_m.append(self.model(stack2[i:min(i + self.gpu_batch, N*B)]))
            
        x_m = torch.cat(x_m)  #torch.Size([6000, 2048, 1, 1]) # masked image1 embedding
        y_m = torch.cat(y_m)  #torch.Size([6000, 2048, 1, 1])  # masked image2 embedding
        
        """ Step 3: Calculating Cosine similarity between the masked image embeddings of one image against the image embedding of another image """
        #Flatten the 4Dtensor into 2D tensor and convert into numpy
        x=x.reshape(x.size(0),-1).cpu().numpy()
        y=y.reshape(y.size(0),-1).cpu().numpy()
        
        x_m=x_m.reshape(x_m.size(0),-1).cpu().numpy() #Convert tensor into numpy 
        y_m=y_m.reshape(y_m.size(0),-1).cpu().numpy() 
        
        sc_x=cosine_similarity(x_m,y) # (6000, 1) similarity scores
        sc_y=cosine_similarity(y_m,x) # (6000, 1) similarity scores
        
        """ Step 4: Calculating Pearsons correlation coefficient between the cosine sim score and all the masks at a specific location """
        # Initialize 
        pearson_corr_x = np.zeros((H1, W1))
        pearson_corr_y = np.zeros((H1, W1))

        for idx in range(H1):
            for jdx in range(W1):
                pearson_corr_x[idx,jdx], _ = pearsonr(sc_x.reshape(-1), self.masks[:,:,idx,jdx].reshape(-1).cpu().detach().numpy()) # returns pearsons correlation and it's p value
                pearson_corr_y[idx,jdx], _ = pearsonr(sc_y.reshape(-1), self.masks[:,:,idx,jdx].reshape(-1).cpu().detach().numpy())
                # print(f"Pearson's Correlation (SciPy): {pearson_corr:.4f}")
        sal_x=pearson_corr_x
        sal_y=pearson_corr_y
        
        return [sal_x, sal_y]
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal

# To process in batches
# def explain_all_batch(data_loader, explainer):
#     n_batch = len(data_loader)
#     b_size = data_loader.batch_size
#     total = n_batch * b_size
#     # Get all predicted labels first
#     target = np.empty(total, 'int64')
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
#         p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
#         target[i * b_size:(i + 1) * b_size] = c
#     image_size = imgs.shape[-2:]
#
#     # Get saliency maps for all images in val loader
#     explanations = np.empty((total, *image_size))
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
#         saliency_maps = explainer(imgs.cuda())
#         explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
#             range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
#     return explanations
