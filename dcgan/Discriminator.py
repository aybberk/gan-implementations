import torch
import torch.nn.functional as F
from helpers import Swish
    
class Discriminator(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        
        
        self.size = size
        def conv(in_channels, out_channels, bn=True, activation=torch.nn.LeakyReLU(0.2)):
            layers = []
            layers.append(torch.nn.Conv2d(in_channels=in_channels, 
                                          out_channels=out_channels,
                                          kernel_size=4, 
                                          stride=2, 
                                          padding=1, 
                                          bias=not bn))
            
            
            if bn:    
                layers.append(torch.nn.BatchNorm2d(out_channels, affine=False))
            
            layers.append(activation)
            return torch.nn.Sequential(*layers)
        
        
        
        self.conv1 = conv(3, 128, bn=False)
        self.conv2 = conv(128, 256)
        self.conv3 = conv(256, 512)
        self.conv4 = conv(512, 1024)
        
        if self.size == [128, 128]:
            self.conv4_128128 = conv(1024,  1024)
        
        
        self.conv5 = torch.nn.Conv2d(1024, 1, 4)
        
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        if self.size == [128, 128]:
            x = self.conv4_128128(x)
        
        
        x = self.conv5(x)
        
        return x.squeeze() 
 
if __name__ is "__main__":
    
    disc = Discriminator()
    x = torch.randn(3, 3, 64, 64)
    x = disc(x)    
    print(x.shape)
#  