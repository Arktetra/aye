import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma = torch.tensor(0.01)):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.W_xh = nn.Parameter(
            torch.randn((num_inputs, num_hiddens)) * sigma
        )
        self.W_hh = nn.Parameter(
            torch.randn((num_hiddens, num_hiddens)) * sigma
        )
        self.b_h = nn.Parameter(
            torch.randn((1, num_hiddens)) * sigma
        )
        
    def forward(self, inputs, state = None):
        if state is None:
            state = torch.zeros((inputs.shape[1], self.num_hiddens), device = inputs.device)
        else:
            state = state 
            
        outputs = []
        for X in inputs:
            state = torch.tanh(
                torch.matmul(X, self.W_xh) 
                + torch.matmul(state, self.W_hh)
                + self.b_h
            )
            outputs.append(state)
            
        return outputs, state