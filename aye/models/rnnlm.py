"""Implementation of a RNN based character level language model."""

from aye.models import BaseModel
from aye.models.foundations.rnn import RNN

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLM(BaseModel):
    """RNN based character level language model."""
    def __init__(self, rnn: RNN, vocab_size, criterion = F.cross_entropy, lr = 0.01):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size 
        self.criterion = criterion
        self.lr = lr
        
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
        )
        self.b_q = nn.Parameter(
            torch.zeros(self.vocab_size)
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
    
    def clip_gradients(self, grad_clip_value):
        params = [p for p in self.parameters()]
        norm = torch.sqrt(sum(p.grad ** 2 for p in params))
        
        if norm > grad_clip_value:
            for param in params:
                param.grad[:] *= grad_clip_value / norm 
                
    def predict(self, prefix, num_preds, vocab, device = None):
        outputs, state = [vocab[prefix[0]]], None 
        
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor(outputs[-1]).to(device)
            embeds = self.one_hot(X)
            print(embeds.shape)
            rnn_outputs, state = self.rnn(embeds.reshape(1, 1, -1), state)      # reshape to (batch_size, num_steps, sequence)
            
            if i < len(prefix) - 1:
                outputs.append(vocab[prefix[i + 1]])
            else:
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(dim = 2).reshape(1)))
        
        return "".join([vocab.idx_to_token[i] for i in outputs])
    
    
        
    