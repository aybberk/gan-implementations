import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
import numpy as np


def torch_to_numpy(img, batch=False):
    
    if batch:
        return img.detach().cpu().numpy().transpose(0, 2, 3, 1)
    else:
        return img.detach().cpu().numpy().transpose(1, 2, 0)
    
def torch_imshow_batch(img_batch, row, col, save=False, name=None):
    if save == True and name is None:
        raise "Specify name for image"
        
    img_batch = img_batch * 0.5 + 0.5;
    img_batch = torch_to_numpy(img_batch, batch=True)
    
    img_size = img_batch.shape[1:]
    img_to_show = np.empty([row, col, *img_size])
    
    
    for r in range(row):
        for c in range(col):
            img_to_show[r, c] = img_batch[r*col + c]
     
    img_to_show = np.concatenate(img_to_show,axis=1)
    img_to_show = np.concatenate(img_to_show,axis=1)
    
    
    fig = plt.figure(figsize=(12,12))
    plt.imshow(img_to_show)
    plt.axis('off')
    plt.show()
    if save:    
        plt.imsave("./imgs/" + name, img_to_show)
        

def torch_imshow(img, save = False, name=None):
     if save == True and name is None:
         raise "Specify name for image"
     img_scaled = img * 0.5 + 0.5
     plt.imshow(torch_to_numpy(img_scaled)); plt.show(); print();
     plt.axis('off')
     if save:    
        plt.imsave("./imgs/" + name, img)


     
     
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
        


        
        
        
        
        
        
        
        
        
        
    