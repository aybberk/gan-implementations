import torch
import torch.nn.functional as F
from helpers import Swish
    
class Discriminator(torch.nn.Module):
    def __init__(self, conditional=True, batchnorm=True):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(0.2)
#        self.activation = Swish()
        self.conditional = conditional
        
        def convlayer(in_channels, 
                      out_channels, 
                      stride, 
                      kernel_size, 
                      padding, 
                      activation = None, 
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
                
            if activation:
                layers.append(activation)
            
            return torch.nn.Sequential(*layers)
        
        activation = torch.nn.LeakyReLU(0.2)
        
        
        self.conv1 = convlayer(3 + conditional, 64,  kernel_size=4, stride=2, padding=1, batchnorm=False, activation=activation)        
        self.conv2 = convlayer(64,              128, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)        
        self.conv3 = convlayer(128,             256, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)        
        self.conv4 = convlayer(256,             512, kernel_size=4, stride=2, padding=1, batchnorm=batchnorm, activation=activation)        
        self.conv5 = convlayer(512,             1  , kernel_size=4, stride=2, padding=1, batchnorm=False)        

        
        
    def forward(self, x_img, x_prior=None):
        
        if self.conditional == True:
            assert not (x_prior is None)
            x = torch.cat([x_img, x_prior], dim=1)

        else:
            x = x_img
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=x.shape[-2:])
        
        x = torch.log(x / (1 - x)) #logit function because our metric is sigmoid BCE with logits
        return x.squeeze() 
 
    
if __name__ == "__main__":
        
    disc = Discriminator()
    x  = torch.randn(5, 3, 256, 256)
    x_ = torch.randn(5, 1, 256, 256)
    x = disc(x, x_)    
    print(x.shape)
      