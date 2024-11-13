from aye import AyeModule
from copy import deepcopy

import torch
import torch.nn as nn 
import torch.nn.functional as F
    
class LeNet5(AyeModule):
    def __init__(self, criterion = F.cross_entropy, act: nn.Module = nn.Sigmoid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2),
            deepcopy(act),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            deepcopy(act),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(in_features = 400, out_features = 120),
            deepcopy(act),
            nn.Linear(in_features = 120, out_features = 84),
            deepcopy(act),
            nn.Linear(in_features = 84, out_features = 10)
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