from aye import AyeModule

import torch
import torch.nn as nn 

class LSTMScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, device, sigma = 0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.device = device
        self.sigma = sigma
        
        self.W_xf, self.W_hf, self.b_f = self.triple()   # Forget gate
        self.W_xi, self.W_hi, self.b_i = self.triple()   # Input gate
        self.W_xc, self.W_hc, self.b_c = self.triple()   # Input node
        self.W_xo, self.W_ho, self.b_o = self.triple()   # Output gate
        
    def init_weights(self, *shape):
        return nn.Parameter(torch.randn(*shape, device = self.device) * self.sigma)
    
    def triple(self):
        return (
            self.init_weights(self.num_inputs, self.num_hiddens),
            self.init_weights(self.num_hiddens, self.num_hiddens),
            nn.Parameter(torch.zeros(self.num_hiddens, device = self.device))
        )
        
    def forward(self, inputs, H_C = None):
        if H_C is None:
            H = torch.zeros((inputs.shape[1], self.num_hiddens), device = inputs.device)
            C = torch.zeros((inputs.shape[1], self.num_hiddens), device = inputs.device)
        else:
            H, C = H_C 
            
        outputs = []
        
        for X in inputs:
            F = torch.sigmoid(
                torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f
            )
            I = torch.sigmoid(
                torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i
            )
            C_tilde = torch.tanh(
                torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c
            )
            O = torch.sigmoid(
                torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o
            )
            
            C = F * C + I * C_tilde
            
            H = O * torch.tanh(C)
            
            outputs.append(H)
        
        return outputs, (H, C)
    
class LSTM(AyeModule):
    def __init__(self, num_inputs, num_hiddens, device = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens 
        self.device = device
        
        self.lstm = nn.LSTM(input_size = num_inputs, hidden_size = num_hiddens)
        
    def forward(self, inputs, H_C = None):
        return self.lstm(inputs, H_C)