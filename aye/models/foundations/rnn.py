import torch
import torch.nn as nn

from aye import AyeModule

class RNNScratch(nn.Module):
    """Implementation of RNN model from scratch."""
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
    
class DeepRNNScratch(nn.Module):
    """Implementation of Deep RNN model from scratch."""
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma = 0.01, device = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers 
        self.sigma = sigma
        self.device = device 
        
        self.rnns = nn.Sequential(*[RNNScratch(num_inputs if i == 0 else num_hiddens, num_hiddens, sigma, device) 
                                    for i in range(num_layers)]
        )
        
    def forward(self, inputs, Hs = None):
        outputs = inputs
        
        if Hs is None:
            Hs = [None] * self.num_layers
            
        for i in range(self.num_layers):
            outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
            outputs = torch.stack(outputs, dim = 0)
            
        return outputs, Hs
    
class BiRNNScratch(AyeModule):
    """Implementation of Bidirectional RNN model from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma = 0.01, device = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens 
        self.sigma = sigma
        self.device = device 
        
        self.f_rnn = RNNScratch(num_inputs, num_hiddens, sigma, device)
        self.b_rnn = RNNScratch(num_inputs, num_hiddens, sigma, device)
        
        self.num_hiddens *= 2
        
    def forward(self, inputs, Hs = None):
        f_H, b_H = Hs if Hs else (None, None)
        
        outputs = []
        
        f_outputs, f_H = self.f_rnn(inputs, f_H)
        b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
        outputs = [torch.cat((f, b), dim = -1) 
                   for f, b in zip(f_outputs, reversed(b_outputs))]
        
        return outputs, (f_H, b_H)
    
class RNN(AyeModule):
    """RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs: int, num_hiddens: int, num_layers: int = 1, bidirectional: bool = False, device = None):
        super().__init__()
        self.num_inputs = num_inputs 
        self.num_hiddens = num_hiddens 
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size = num_inputs, hidden_size = num_hiddens, 
            num_layers = num_layers, bidirectional = bidirectional,
            device = device,
        )
        
    def forward(self, inputs, state = None):
        return self.rnn(inputs, state)
        