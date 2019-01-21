###############################################################################
#### ORIGINAL CYCLEGAN
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
import itertools

NUM_ITERATIONS = 40000
BATCH_SIZE = 2
K = 1
LOG_EVERY = 20
DEVICE = "cuda"
ADAM_LR = 2e-4
ADAM_BETAS = [0.5, 0.999]
LAMBDA_CYC = 10


 
X_transform = transforms.Compose([
        transforms.Resize([256, 256]), ###########TODO######BUNU SONRAYA AL DENE
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x:2*x - 1),
        transforms.Lambda(lambda x: x.to(DEVICE)),
        ])


Y_transform = transforms.Compose([
        transforms.Resize([256, 256]), 
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x:2*x - 1),
        transforms.Lambda(lambda x: x.to(DEVICE)),
        ])


mnist_dataset_path     = '/home/ayb/DATASETS/mnist'
cats_dataset_path      = '/home/ayb/DATASETS/cats'
dogs_dataset_path      = '/home/ayb/DATASETS/dogs'
pokemon_a_dataset_path = '/home/ayb/DATASETS/pokemon/pokemon-a'
pokemon_b_dataset_path = '/home/ayb/DATASETS/pokemon/pokemon-b'
celeb_dataset_path     = '/home/ayb/DATASETS/celeb'
horses_dataset_path    = '/home/ayb/DATASETS/horses'
zebras_dataset_path    = '/home/ayb/DATASETS/zebras'

toy_dataset_path       = '/home/ayb/DATASETS/toy'
toy2_dataset_path      = '/home/ayb/DATASETS/toy2'

#mnist_dataset = datasets.MNIST(, train=True, download=True, transform=X_transform)



#CHANGE THESE TO CHANGE DATASET
X_dataset_path = celeb_dataset_path
Y_dataset_path = celeb_dataset_path



X_dataset = datasets.ImageFolder(root=X_dataset_path, transform=X_transform)
Y_dataset = datasets.ImageFolder(root=Y_dataset_path, transform=Y_transform)


X_loader = torch.utils.data.DataLoader(X_dataset,
                                       batch_size=BATCH_SIZE, 
                                       shuffle=True,
                                       drop_last=True)


Y_loader = torch.utils.data.DataLoader(Y_dataset,
                                       batch_size=BATCH_SIZE, 
                                       shuffle=True,
                                       drop_last=True)


# G: X -> Y
G = Generator().to(DEVICE)

# F: Y -> X
F = Generator().to(DEVICE)


DY = Discriminator().to(DEVICE)
DX = Discriminator().to(DEVICE)



BCE = torch.nn.BCEWithLogitsLoss()
LS  = torch.nn.MSELoss()

L1 = torch.nn.L1Loss()


GF_parameters = itertools.chain(G.parameters(), F.parameters())
GF_optimizer  = torch.optim.Adam(GF_parameters,  lr=ADAM_LR, betas=ADAM_BETAS)

D_parameters  = itertools.chain(DX.parameters(), DY.parameters())
D_optimizer = torch.optim.Adam(D_parameters, lr=ADAM_LR, betas=ADAM_BETAS)



iterations = 0
last_log_time = time.time()
while iterations < NUM_ITERATIONS or True:
    for (x, _), (y, _)  in zip(X_loader, Y_loader):
        
        real_label = torch.ones(BATCH_SIZE).to(DEVICE)
        fake_label = torch.zeros(BATCH_SIZE).to(DEVICE)
        
        #D step
        G_x   = G(x)
        F_y   = F(y)

        DY_real = DY(y)
        DY_fake = DY(G_x)
        
        DX_real = DX(x)
        DX_fake = DX(F_y)
                
        DY_real_loss = LS(DY_real, real_label)
        DY_fake_loss = LS(DY_fake, fake_label)
        DY_loss = (DY_real_loss + DY_fake_loss) / 2
        
        DX_real_loss = LS(DX_real, real_label)
        DX_fake_loss = LS(DX_fake, fake_label)
        DX_loss = (DX_real_loss + DX_fake_loss) / 2
        
        total_D_loss =  DX_loss + DY_loss
                    
        D_optimizer.zero_grad()
        total_D_loss.backward()
        D_optimizer.step()


        
        
        #G&F step
        G_x   = G(x)
        F_y   = F(y)
        FoG_x = F(G_x)
        GoF_y = G(F_y)
        
        
        cycle_loss = L1(FoG_x, x) + L1(GoF_y, y)
        

        DY_fake = DY(G_x)
        G_adversarial_loss = LS(DY_fake, real_label)
        
        DX_fake = DX(F_y)
        F_adversarial_loss = LS(DX_fake, real_label)
        
        total_GF_loss = G_adversarial_loss + F_adversarial_loss + LAMBDA_CYC * cycle_loss
        
        GF_optimizer.zero_grad()            
        total_GF_loss.backward()
        GF_optimizer.step()
        
        

        
        iterations += 1   
        if iterations % LOG_EVERY == 0:
            
            secs_remaining = (time.time() - last_log_time) * (NUM_ITERATIONS - iterations) / LOG_EVERY
            last_log_time = time.time()
            
            secs_ = secs_remaining % 60
            mins  = (secs_remaining - secs_) / 60
            mins_ = mins % 60
            hours = (mins - mins_) / 60
            
            
            torch_imshow(torch.cat([x[0],G_x[0], FoG_x[0]], dim=2))
            torch_imshow(torch.cat([y[0],F_y[0], GoF_y[0]], dim=2))
            
            
            print("Iterations: ", iterations)
            print("DX_loss:", DX_loss.detach().cpu().numpy())
            print("DY_loss:", DY_loss.detach().cpu().numpy())
            print("---")            
            print("G_loss:", G_adversarial_loss.detach().cpu().numpy())
            print("F_loss:", F_adversarial_loss.detach().cpu().numpy())
            print("---")
            print("Cycle_loss:", cycle_loss.detach().cpu().numpy())            
            
            
            print("Time remaining: {} hours {} mins {} seconds. (%{})".format(int(hours), int(mins_), int(secs_), int(iterations / NUM_ITERATIONS * 100)))

        
