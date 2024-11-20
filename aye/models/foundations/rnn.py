import torch
import torch.nn as nn

from aye import AyeModule

class RNNScratch(nn.Module):
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma = 0.01, device = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        self.W_xh = nn.Parameter(
            torch.randn((num_inputs, num_hiddens), device = device) * sigma,
            
        )
        self.W_hh = nn.Parameter(
            torch.randn((num_hiddens, num_hiddens), device = device) * sigma,
        )
        self.b_h = nn.Parameter(
            torch.randn((1, num_hiddens), device = device) * sigma,
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
    
class RNN(AyeModule):
    """RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, sigma = 0.01, device = None):
        super().__init__()
        self.num_inputs = num_inputs 
        self.num_hiddens = num_hiddens 
        
        self.rnn = nn.RNN(input_size = num_inputs, hidden_size = num_hiddens, device = device)
        
    def forward(self, inputs, state = None):
        return self.rnn(inputs, state)