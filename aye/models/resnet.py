from aye.models import BaseModel 

import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, out_channels, skip_connection: bool = False, strides = 1):
        super().__init__()
        self.skip_connection = skip_connection
        self.residual = nn.Sequential(
            nn.LazyConv2d(out_channels = out_channels, kernel_size = 3, stride = strides, padding = 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels = out_channels, kernel_size = 3, padding = 1),
            nn.LazyBatchNorm2d()
        )
        
        if self.skip_connection:
            self.skip_layer = nn.LazyConv2d(
                out_channels = out_channels, kernel_size = 1, stride = strides
            )
        else:
            self.skip_layer = None
            
    def forward(self, x):
        y = self.residual(x)
        if self.skip_connection:
            x = self.skip_layer(x)
        
        return F.relu(y + x)
    
class ResNet(BaseModel):
    def __init__(self, arch, lr = 1e-3, criterion: nn = F.cross_entropy, num_classes = 10):
        super().__init__()
        self.arch = arch
        self.lr = lr
        self.criterion = criterion
        self.num_classes = num_classes
        
        self.net = nn.Sequential(self.b1())
        
        for i, b in enumerate(arch):
            self.net.add_module(f"b{i + 2}", self.block(*b, first_block = (i == 0)))
        
        self.net.add_module("last", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))
        
    def forward(self, x):
        return self.net(x)
        
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.LazyBatchNorm2d(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
    def block(self, num_residuals, out_channels, first_block = False):
        blocks = []
        
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blocks.append(Residual(out_channels = out_channels, skip_connection = True, strides = 2))
            else:
                blocks.append(Residual(out_channels = out_channels))
                
        return nn.Sequential(*blocks)
    
class ResNet18(ResNet):
    def __init__(self, lr = 1e-3, criterion: nn = F.cross_entropy, num_classes = 10): 
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)), lr, criterion, num_classes)