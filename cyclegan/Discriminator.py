import torch
import torch.nn.functional as F
    
class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        def C(in_channels, 
              out_channels,
              instance_norm=True):
              
            layers = []
            
            layers.append(torch.nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=4,
                                          stride=2,
                                          padding=1))
            
            layers.append(torch.nn.InstanceNorm2d(out_channels))
            
            layers.append(torch.nn.LeakyReLU(0.2))
                      
            return torch.nn.Sequential(*layers)
        
        
        self.block1 = C(3,   64, instance_norm=False)
        self.block2 = C(64,  128)
        self.block3 = C(128, 256)
        self.block4 = C(256, 512)
        #
#        self.block5 = C(512, 512)
#        self.block6 = C(512, 512)
        #
        self.blocklast = torch.nn.Conv2d(512, 1, kernel_size=4)
        
    def forward(self, x_img, x_prior=None):
        

        x = x_img
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        #
#        x = self.block5(x)
#        x = self.block6(x)
        #
        x = self.blocklast(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[-2:])
        
        
        return x.squeeze() 
 
    
if __name__ == "__main__":
        
    disc = Discriminator()
    x  = torch.randn(5, 3, 256, 256)
    x_ = torch.randn(5, 1, 256, 256)
    x = disc(x, x_)    
    print(x.shape)
    
    
    
    
    
      