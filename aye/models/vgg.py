from aye import AyeModule
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def vgg_block(num_convs, out_channels):
    layers = []
    
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels = out_channels, kernel_size = 3, padding = 1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    
    return nn.Sequential(*layers)

class VGG(AyeModule):
    """
    Create a VGG model.
    
    Args:
        arch (List[Tuple[int]]): List of number of convolution layers and output channels in each VGG block.
        num_classes (int, optional): Number of classes. Defaults to 10.
        criterion (nn.functional, optional): Objective function. Defaults to F.cross_entropy.
    
    # Example
    
    A VGG-11 can be created as:
    
    >>> from aye.models import VGG

    >>> model = VGG(
        arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    )

    >>> model.summary(x_shape = (16, 1, 224, 224))
    """
    def __init__(self, arch: List[Tuple[int]], num_classes: int = 10, criterion: nn.functional = F.cross_entropy):
        super().__init__()
        self.arch = arch 
        self.criterion = criterion
        self.num_classes = num_classes
        
        conv_blocks = []
        
        for (num_convs, out_channels) in arch:
            conv_blocks.append(vgg_block(num_convs, out_channels))
            
        self.net = nn.Sequential(
            *conv_blocks,
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, x):
        return self.net(x) 

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        self.logits = self(x)
        return self.criterion(self.logits, y)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def configure_optimizers(self, lr = 1e-3):
        return torch.optim.Adam(params = self.parameters(), lr = lr)