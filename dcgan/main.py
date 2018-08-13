###############################################################################
#### ORIGINAL DCGAN   
#### IMPLEMENTATION
###############################################################################

import torch
from Generator import Generator
from Discriminator import Discriminator
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.utils.data
from helpers import torch_imshow


NUM_ITERATIONS = 500000
BATCH_SIZE = 16
LATENT_DIM = 100
K = 1
LOG_EVERY = 15
NOISE_STD = 1 
LABEL_SMOOTHING = True
DEVICE = "cuda"


 
data_transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.RandomHorizontalFlip(),
#            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x:2*x - 1),
            ])
    
    
mnist_dataset = datasets.MNIST('/home/ayb/DATASETS/mnist', train=True, download=True, transform=data_transform)
cat_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/CAT_DATASET', transform=data_transform)
pokemon_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/pokemon/pokemon-b', transform=data_transform)
celeb_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/celeb', transform=data_transform)    
toy_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/toy', transform=data_transform)
    

### CHANGE FIRST ARGUMENT TO CHANGE DATASET ###
data_loader = torch.utils.data.DataLoader(celeb_dataset,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True,
                                          drop_last=True)



G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
BCE = torch.nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=[0.5, 0.999])
g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=[0.5, 0.999])



#try:
#    iterations_per_epoch = len(data_loader.dataset.imgs) // data_loader.batch_size
#except:
#    iterations_per_epoch = len(data_loader.dataset.train_data) // data_loader.batch_size
    
iterations = 0
while iterations < NUM_ITERATIONS:
#    print("EPOCH: {}".format(epoch))
    for x_data, target in data_loader:
            
        x_data = x_data.to(DEVICE)
        #D step
        for n in range(K):
            z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE) * NOISE_STD
            x_gen  = G(z)
       
                
            D_real   = D(x_data)
            real_label = torch.ones(BATCH_SIZE).to(DEVICE) * (0.9 if LABEL_SMOOTHING else 1.0)
            
            D_fake   = D(x_gen)
            fake_label = torch.zeros(BATCH_SIZE).to(DEVICE)
    
            real_loss = BCE(D_real, real_label)
            fake_loss = BCE(D_fake, fake_label)
            
            d_loss = real_loss + fake_loss
    
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
        
        #G step
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE) * NOISE_STD
        x_gen  = G(z)
        D_gen = D(x_gen)
        real_label = torch.ones(BATCH_SIZE).to(DEVICE)
        g_loss = BCE(D_gen, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        
        if iterations % LOG_EVERY == 0:
            x_gen_show = G(torch.randn(1, LATENT_DIM).to(DEVICE))
            torch_imshow(x_gen_show[0])
            print("Iterations: ", iterations)
            print("G_loss:", g_loss.detach().cpu().numpy())
            print("D_loss:", d_loss.detach().cpu().numpy())
#            print("Epoch: {:0.2f}".format(iterations / iterations_per_epoch))

        iterations += 1   
        
