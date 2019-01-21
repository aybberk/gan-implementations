import torch
import torch.nn.functional as F
from helpers import Swish

class Generator(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
#        
        
        self.size = size
        self.activation = torch.nn.ReLU()
#       self.activation = Swish()
        
        def convT(in_channels, out_channels, stride=2, padding=1, bn=True, activation=torch.nn.ReLU()):
            layers = []
            layers.append(torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                                   out_channels=out_channels,
                                                   kernel_size=4, 
                                                   stride=stride, 
                                                   padding=padding, 
                                                   bias=not bn))
            
            
            if bn:    
                layers.append(torch.nn.BatchNorm2d(out_channels, affine=False))
            
            layers.append(activation)
            
            return torch.nn.Sequential(*layers)
        
        
        self.proj   = convT(100, 1024, stride=1, padding=0)
        self.convT1 = convT(1024, 512)
        self.convT2 = convT(512,  256)
        self.convT3 = convT(256,  128)
        
        if self.size == [128, 128]:
            self.convT3_128128 = convT(128,  128)
        
        
        self.convT4 = convT(128,  3, bn=False, activation=torch.nn.Tanh())
        

        
    def forward(self, x):
        
        x = x.view(-1, 100, 1, 1)
        x = self.proj(x)
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        
        if self.size == [128, 128]:
            x = self.convT3_128128(x)
        
        x = self.convT4(x)
        return x




if __name__ is "__main__":

    gen = Generator()
    x = torch.randn(3, 100)
    x = gen(x)    
    print(x.shape)