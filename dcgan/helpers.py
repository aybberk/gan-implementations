import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data

def torch_imshow(img):
     img_scaled = img * 0.5 + 0.5
     plt.imshow(img_scaled.detach().cpu().numpy().transpose(1, 2, 0)); plt.show(); print()
            
     
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
    
def find_closest(dataset, img):
    
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1)
    
    
    min_dist = 999999999
    closest = None
    img = img.to("cpu")
    
    iters = 0
    for x, target in data_loader:
        iters += 1
        x = x[0]
        dist = torch.mean((x - img)**2)
        if dist < min_dist:
            closest = x
            min_dist = dist
            torch_imshow(img)
            torch_imshow(closest)
            print(iters / len(dataset.imgs))
    
    return closest
        


        
        
        
        
        
        
        
        
        
        
    