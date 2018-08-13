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
from helpers import stack_with_edges, torch_imshow
import time


NUM_ITERATIONS = 40000
BATCH_SIZE = 8
LATENT_DIM = 100
K = 1
LOG_EVERY = 20
LABEL_SMOOTHING = True
DEVICE = "cuda"

SKIP_CONNECTIONS = False
LAMBDA = 100
CONDITIONAL_D = True
BATCH_NORM = True


 
data_transform = transforms.Compose([
            transforms.Resize([256, 256]), ###########TODO######BUNU SONRAYA AL DENE
            transforms.RandomHorizontalFlip(),
#            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x:stack_with_edges(x, sigma=1.5)),
            transforms.Lambda(lambda x:2*x - 1),
            transforms.Lambda(lambda x: x.to(DEVICE)),
            transforms.Lambda(lambda x: (x[:1], x[1:]))
            ])
    
    
mnist_dataset = datasets.MNIST('/home/ayb/DATASETS/mnist', train=True, download=True, transform=data_transform)
cat_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/CAT_DATASET', transform=data_transform)
pokemon_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/pokemon/pokemon-b', transform=data_transform)
celeb_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/celeb', transform=data_transform)    
toy_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/toy', transform=data_transform)
    

### CHANGE FIRST ARGUMENT TO CHANGE DATASET ###
data_loader = torch.utils.data.DataLoader(pokemon_dataset,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True,
                                          drop_last=True)



G = Generator(skip=SKIP_CONNECTIONS, batchnorm=BATCH_NORM).to(DEVICE)
D = Discriminator(conditional=CONDITIONAL_D).to(DEVICE)
BCE = torch.nn.BCEWithLogitsLoss()
L1 = torch.nn.L1Loss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=[0.5, 0.999])
g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=[0.5, 0.999])



    
iterations = 0
last_log_time = time.time()
while iterations < NUM_ITERATIONS or True:
    for x, target in data_loader:
        
        
        x_prior, x_data = x
        #D step
        for n in range(K):
            x_gen  = G(x_prior)
       
                
            D_real   = D(x_data, x_prior)
            real_label = torch.ones(BATCH_SIZE).to(DEVICE) * (0.9 if LABEL_SMOOTHING else 1.0)
            
            D_fake   = D(x_gen, x_prior)
            fake_label = torch.zeros(BATCH_SIZE).to(DEVICE)
    
            real_loss = BCE(D_real, real_label)
            fake_loss = BCE(D_fake, fake_label)
            
            loss_D = (real_loss + fake_loss) / 2
    
            
            d_optimizer.zero_grad()
            loss_D.backward()
            d_optimizer.step()
        
        #G step
        x_gen  = G(x_prior)
        D_fake = D(x_gen, x_prior)
        real_label = torch.ones(BATCH_SIZE).to(DEVICE)
        loss_G = BCE(D_fake, real_label) + LAMBDA * L1(x_gen, x_data)
        g_optimizer.zero_grad()
        loss_G.backward()
        g_optimizer.step()
        
        
        iterations += 1   
        if iterations % LOG_EVERY == 0:
            x_gen_show = G(x_prior)
            torch_imshow(x_data[0])
            torch_imshow(torch.cat([torch.cat([x_prior[0]]*3), x_gen_show[0]], dim=2))
            
            secs_remaining = (time.time() - last_log_time) * (NUM_ITERATIONS - iterations) / LOG_EVERY
            last_log_time = time.time()
            
            secs_ = secs_remaining % 60
            mins  = (secs_remaining - secs_) / 60
            mins_ = mins % 60
            hours = (mins - mins_) / 60
            
            
            
            print("Iterations: ", iterations)
            print("G_loss:", loss_G.detach().cpu().numpy())
            print("D_loss:", loss_D.detach().cpu().numpy())
            print("Time remaining: {} hours {} mins {} seconds. (%{})".format(int(hours), int(mins_), int(secs_), int(iterations / NUM_ITERATIONS * 100)))
