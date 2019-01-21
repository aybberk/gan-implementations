import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
from skimage import feature
import numpy as np

def torch_imshow(img):
    img = img * 0.5 + 0.5
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    plt.imshow(img, cmap="gray")
    plt.show(); print()
     
class JointDataset(torch.utils.data.Dataset):
    def __init__(self, d1, d2):
        super().__init__()
        assert len(d1) == len(d2)
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1)
        
    def __getitem__(self, index):
        return self.d1[index], self.d2[index]
    
    
class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, N, dim):
        super().__init__()
        self.data = torch.randn(N, dim)
        self.N = N
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        return self.data[index], 0
    
    
class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * F.sigmoid(x)
   

def img_to_edges(x, sigma=1):
    x_grey = x.numpy().mean(axis=0)
    x_edges = feature.canny(x_grey, sigma=sigma).astype(np.float32)
    x_edges = torch.tensor(np.expand_dims(x_edges, axis=0))
    return x_edges
    
def stack_with_edges(x, sigma=1):
    x_edges = img_to_edges(x, sigma=sigma)
    return torch.cat([1 - x_edges, x])
        
        
        
        
        
        
        
        
    