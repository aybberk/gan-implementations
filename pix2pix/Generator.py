import torch
import torch.nn.functional as F
from helpers import Swish
from functools import partial

class Generator(torch.nn.Module):
    def __init__(self, skip=True, batchnorm=True):
        super().__init__()
        self.skip = skip
        
        def convlayer(in_channels, 
                      out_channels, 
                      stride, 
                      kernel_size, 
                      padding, 
                      activation, 
                      batchnorm = False, 
                      dropout_ratio = 0.0, 
                      transpose=False):
            
            layers = []
            
            if transpose:
                conv = torch.nn.ConvTranspose2d
            else:
                conv = torch.nn.Conv2d
                
            layers.append(conv(in_channels = in_channels,
                               out_channels = out_channels, 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding))
            
            if batchnorm:
                layers.append(torch.nn.BatchNorm2d(out_channels))
                
            if dropout_ratio > 0:
                layers.append(torch.nn.Dropout2d(dropout_ratio))
                
            layers.append(activation)
            
            return torch.nn.Sequential(*layers)
        
        
        activation  = torch.nn.LeakyReLU(0.2)
#       activation = Swish()
        
        
        self.enc1 = convlayer(1,   64,  kernel_size=4, stride=2, padding=1, batchnorm=False,     activation=activation)        
        self.enc2 = convlayer(64,  128, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)                
        self.enc3 = convlayer(128, 256, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)        
        self.enc4 = convlayer(256, 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)        
        self.enc5 = convlayer(512, 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)        
        self.enc6 = convlayer(512, 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)        
        self.enc7 = convlayer(512, 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)
        self.enc8 = convlayer(512, 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)
        
#---------------------------------------------------------------------------------#
        activation = torch.nn.ReLU()
        
        self.dec1 = convlayer(512,                   512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, dropout_ratio=0.5, activation=activation, transpose=True)
        self.dec2 = convlayer(512 * (1 + self.skip), 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, dropout_ratio=0.5, activation=activation, transpose=True)
        self.dec3 = convlayer(512 * (1 + self.skip), 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, dropout_ratio=0.5, activation=activation, transpose=True)
        self.dec4 = convlayer(512 * (1 + self.skip), 512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation, transpose=True)
        self.dec5 = convlayer(512 * (1 + self.skip), 256, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation, transpose=True)
        self.dec6 = convlayer(256 * (1 + self.skip), 128, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation, transpose=True)
        self.dec7 = convlayer(128 * (1 + self.skip), 64,  kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation, transpose=True)
        self.dec8 = convlayer(64  * (1 + self.skip), 3  , kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=torch.nn.Tanh(),  transpose=True)
        
        
    def forward(self, x_prior):

#       ENCODER
        x = x_prior
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)

#       DECODER
        if self.skip == True:
            x = self.dec1(x8)
            x = self.dec2(torch.cat([x, x7], dim=1))
            x = self.dec3(torch.cat([x, x6], dim=1))
            x = self.dec4(torch.cat([x, x5], dim=1))
            x = self.dec5(torch.cat([x, x4], dim=1))
            x = self.dec6(torch.cat([x, x3], dim=1))
            x = self.dec7(torch.cat([x, x2], dim=1))
            x = self.dec8(torch.cat([x, x1], dim=1))
        else:
            x = self.dec1(x8)
            x = self.dec2(x)
            x = self.dec3(x)
            x = self.dec4(x)
            x = self.dec5(x)
            x = self.dec6(x)
            x = self.dec7(x)
            x = self.dec8(x)        
        
        return x
    
    

        
if __name__ == "__main__":
    gen = Generator(skip=True, batchnorm=False)
    x = torch.randn(5, 1, 256, 256)
    x = gen(x)    
    print(x.shape)