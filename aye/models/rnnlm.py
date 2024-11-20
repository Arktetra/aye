"""Implementation of a RNN based character level language model."""

from aye.models import BaseModel
from aye.models.foundations.rnn import RNN

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLMScratch(BaseModel):
    """RNN based character level language model implemented from scratch."""
    def __init__(self, rnn: RNN, vocab_size, criterion = F.cross_entropy, lr = 1e-3, device = None):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size 
        self.criterion = criterion
        self.lr = lr
        self.device = device
        
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn((self.rnn.num_hiddens, self.vocab_size), device = self.device) * self.rnn.sigma
        )
        self.b_q = nn.Parameter(
            torch.zeros(self.vocab_size, device = self.device)
        )
        
    def output_layer(self, rnn_outputs):
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return torch.stack(outputs, dim = 0)    
    
    def forward(self, x, state = None):
        embeds = self.one_hot(x)
        rnn_outputs, _ = self.rnn(embeds, state)
        return self.output_layer(rnn_outputs)
    
    def __shared_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        self.logits = self(x)
        return self.criterion(self.logits, self.one_hot(y))
        
    def training_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx)
    
    def one_hot(self, x):
        return F.one_hot(x, self.vocab_size).type(torch.float32)
                
    def predict(self, prefix, num_preds, vocab, device = None):
        outputs, state = [vocab[prefix[0]]], None 
        
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor(outputs[-1]).to(device)
            embeds = self.one_hot(X).reshape(1, 1, -1)      # reshape to (batch_size, num_steps, sequence)
            rnn_outputs, state = self.rnn(embeds, state)      
            
            if i < len(prefix) - 1:
                outputs.append(vocab[prefix[i + 1]])
            else:
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(dim = 2).reshape(1)))
        
        return "".join([vocab.idx_to_token[i] for i in outputs])
    
class RNNLM(RNNLMScratch):
    """RNN-based character level language model implemented with high-level APIs."""
    
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size, device = self.device)
        
    def output_layer(self, hiddens):
        return self.linear(hiddens)