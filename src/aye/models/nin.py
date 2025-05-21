from aye.models import BaseModel

import torch.nn as nn
import torch.nn.functional as F

def nin_block(out_channels: int, kernel_size: int, stride: int, padding: int):
    """
    Returns a Network in Network (NiN) block consisting of a convolution layer of 
    the supplied parameters followed by 1 x 1 convolution layers.
    """
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, stride, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size = 1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size = 1), nn.ReLU()
    )
    
class NiN(BaseModel):
    """
    Create a Network in Network (NiN) model.
    """
    def __init__(self, lr = 1e-3, criterion: F = F.cross_entropy, num_classes: int = 10):
        super().__init__()
        self.criterion = criterion
        self.num_classes = num_classes 
        self.lr = lr
        
        self.net = nn.Sequential(
            nin_block(out_channels = 96, kernel_size = 11, stride = 4, padding = 0),
            nn.MaxPool2d(3, stride = 2),
            nin_block(out_channels = 256, kernel_size = 5, stride = 1, padding = 2),
            nn.MaxPool2d(3, stride = 2),
            nin_block(384, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(3, stride = 2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
                
    def forward(self, x):
        return self.net(x) 
