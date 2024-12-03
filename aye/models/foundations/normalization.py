import torch 
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w = nn.Parameter(torch.ones(config.d_model))
        self.b = nn.Parameter(torch.zeros(config.d_model))
        
    def forward(self, residual):
        mean = residual.mean(dim = -1, keepdim = True)
        var = residual.var(dim = -1, keepdim = True, unbiased = False)
        residual = (residual - mean) / (var + self.config.layer_norm_eps).sqrt()
        return residual * self.w + self.b