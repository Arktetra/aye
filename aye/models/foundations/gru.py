from aye import AyeModule

import torch
import torch.nn as nn 

class GRUScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma = 0.01, device = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens 
        self.sigma = sigma
        self.device = device 
        
        self.W_xr, self.W_hr, self.b_r = self.triple()      # Reset gate
        self.W_xz, self.W_hz, self.b_z = self.triple()      # Update gate
        self.W_xh, self.W_hh, self.b_h = self.triple()      # Candidate Hidden State
        
    def init_weights(self, *shape):
        return nn.Parameter(
            torch.randn(*shape, device = self.device)
        )
        
    def triple(self):
        return (
            self.init_weights((self.num_inputs, self.num_hiddens)),
            self.init_weights((self.num_hiddens, self.num_hiddens)),
            nn.Parameter(torch.zeros(self.num_hiddens, device = self.device))
        )
        
    def forward(self, inputs, H = None):
        if H is None:
            H = torch.zeros((inputs.shape[1], self.num_hiddens), device = self.device)
        
        outputs = []
        
        for X in inputs:
            R = torch.sigmoid(
                torch.matmul(X, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r
            )
            
            Z = torch.sigmoid(
                torch.matmul(X, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z
            )
            
            H_tilde = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(H * R, self.W_hh) + self.b_h
            )
            
            H = Z * H + (1 - R) * H_tilde 
            
            outputs.append(H)
            
        return outputs, H
    
class GRU(AyeModule):
    def __init__(self, num_inputs, num_hiddens, sigma = 0.01, device = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens 
        self.sigma = sigma
        self.device = device 
        
        self.gru = nn.GRU(input_size = num_inputs, hidden_size = num_hiddens, device = self.device)
        
    def forward(self, X, H = None):
        return self.gru(X, H)