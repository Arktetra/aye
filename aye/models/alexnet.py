from aye import AyeModule
from copy import deepcopy

import torch
import torch.nn as nn 
import torch.nn.functional as F 

class AlexNet(AyeModule):
    def __init__(self, criterion = F.cross_entropy, act = nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4),
            deepcopy(act),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2),
            deepcopy(act),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1),
            deepcopy(act),
            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1),
            deepcopy(act),
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1),
            deepcopy(act),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Flatten(),
            nn.Linear(in_features = 6400, out_features = 4096),
            deepcopy(act),
            nn.Linear(in_features = 4096, out_features = 4096),
            deepcopy(act),
            nn.Linear(in_features = 4096, out_features = 1000),
        )
        
        self.criterion = criterion
        
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