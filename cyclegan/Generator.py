import torch
import torch.nn.functional as F
from helpers import Swish
from functools import partial

class Generator(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        
        
        def c7s1(in_channels, 
                 out_channels,
                 activation=torch.nn.ReLU()):
            
            layers = []
            
            layers.append(torch.nn.ReflectionPad2d(3)) #TODO burayi degis de dene
            
            layers.append(torch.nn.Conv2d(in_channels=in_channels, 
                                          out_channels=out_channels, 
                                          kernel_size=7, 
                                          stride=1))
            
            layers.append(torch.nn.InstanceNorm2d(out_channels))
            
            layers.append(activation)

            return torch.nn.Sequential(*layers)            
        
        
        def d(in_channels, 
              out_channels,
              ):
            
            layers = []
            
            layers.append(torch.nn.ReflectionPad2d(1))
            
            layers.append(torch.nn.Conv2d(in_channels=in_channels, 
                                          out_channels=out_channels, 
                                          kernel_size=3, 
                                          stride=2))
            
            layers.append(torch.nn.InstanceNorm2d(out_channels))
            
            layers.append(torch.nn.ReLU())
            
            return torch.nn.Sequential(*layers)
        
        
                        
        
        
        class R(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                
                self.conv1 = torch.nn.Sequential(
                        torch.nn.ReflectionPad2d(1),
                        torch.nn.Conv2d(in_channels, 
                                        out_channels,
                                        kernel_size=3,
                                        stride=1))
                
                
                self.conv2 = torch.nn.Sequential(
                        torch.nn.ReflectionPad2d(1),
                        torch.nn.Conv2d(in_channels, 
                                        out_channels,
                                        kernel_size=3,
                                        stride=1))

                        
            def forward(self, x_in):
                x = torch.nn.functional.relu(self.conv1(x_in))
                x = torch.nn.functional.relu(self.conv2(x) + x_in)
                return x
        

                
        
        def u(in_channels,
              out_channels):
            layers = []
            layers.append(torch.nn.ConvTranspose2d(in_channels, 
                                                   out_channels, 
                                                   kernel_size=4, 
                                                   stride=2, 
                                                   padding=1))
            
            
            layers.append(torch.nn.InstanceNorm2d(out_channels))
            
            layers.append(torch.nn.ReLU())
            
            return torch.nn.Sequential(*layers)


        
        self.block1 = c7s1(3,   32)     #c7s1-32
        self.block2 =    d(32,  64)     #d64
        self.block3 =    d(64,  128)    #d128
        self.block4 =    torch.nn.Sequential(*[R(128, 128) for n in range(9)]) #9 residual blocks(R128)
        self.block5 =    u(128, 64)     #u64
        self.block6 =    u(64,  32)     #u32
        self.block7 = c7s1(32,  3, activation=torch.nn.Tanh())      #c7s1-3
        
        
        
        
                
        
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x

    

        
if __name__ == "__main__":
    gen = Generator()
    x = torch.randn(5, 3, 256, 256)
    x = gen(x)    
    print(x.shape)