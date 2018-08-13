import torch
from torchvision import datasets, transforms
from helpers import stack_with_edges, torch_imshow
from Generator import Generator

DEVICE = "cuda"



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
Y_dataset_path = toy_dataset_path



X_dataset = datasets.ImageFolder(root=X_dataset_path, transform=X_transform)
Y_dataset = datasets.ImageFolder(root=Y_dataset_path, transform=Y_transform)


X_loader = torch.utils.data.DataLoader(X_dataset,
                                       batch_size=2, 
                                       shuffle=True,
                                       drop_last=True)


Y_loader = torch.utils.data.DataLoader(Y_dataset,
                                       batch_size=2, 
                                       shuffle=True,
                                       drop_last=True)



G = Generator().to(DEVICE)
G.load_state_dict(torch.load("../saved-models/cyclegan/G"))

F = Generator().to(DEVICE)
F.load_state_dict(torch.load("../saved-models/cyclegan/F"))

 

for (x, _), (y, _)  in zip(X_loader, Y_loader):
        
    x_gen_show = F(y)
    torch_imshow(torch.cat([x_gen_show[0]], dim=2))
        
