import torch
from torchvision import datasets, transforms
from helpers import stack_with_edges, torch_imshow
from Generator import Generator

DEVICE = "cuda"
G = Generator(skip=True, batchnorm=False).to(DEVICE)
G.load_state_dict(torch.load("../saved-models/pix2pix/pokemon-edge-to-real-G"))



 
data_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
#            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: stack_with_edges(x, sigma=1.)),
            transforms.Lambda(lambda x:2*x - 1),
            transforms.Lambda(lambda x: x.to(DEVICE)),
            transforms.Lambda(lambda x: (x[:1], x[1:]))
            ])

    
mnist_dataset = datasets.MNIST('/home/ayb/DATASETS/mnist', train=True, download=True, transform=data_transform)
cat_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/CAT_DATASET', transform=data_transform)
pokemon_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/pokemon/pokemon-a', transform=data_transform)
celeb_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/celeb', transform=data_transform)    
toy_dataset = datasets.ImageFolder(root='/home/ayb/DATASETS/toy', transform=data_transform)
    

### CHANGE FIRST ARGUMENT TO CHANGE DATASET ###
data_loader = torch.utils.data.DataLoader(toy_dataset,
                                          batch_size=2, 
                                          shuffle=True,
                                          drop_last=True)

for x, target in data_loader:
    
    x_prior, x_data = x
    x_gen_show = G(x_prior)
    torch_imshow(x_prior[0])
    torch_imshow(torch.cat([x_data[0], x_gen_show[0]], dim=2))
        
