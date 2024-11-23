from aye.models.foundations.encoder_decoder import Encoder, Decoder, EncoderDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqEncoder(Encoder):
    """Encoder for sequence to sequence model."""
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        num_hiddens: int,
        num_layers: int, 
        device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size 
        self.embed_size = embed_size 
        self.num_hiddens = num_hiddens
        self.device = device
        
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)
        self.rnn = nn.GRU(input_size = embed_size, hidden_size = num_hiddens, num_layers = num_layers)
        
    def forward(self, X: torch.Tensor, *args):              # X.shape = (batch_size, time_steps)
        embeds = self.embedding(X.t().type(torch.int64))    # embeds.shape = (time_steps, batch_size, embed_size)
        output, state = self.rnn(embeds)                    # output.shape = (time_steps, batch_size, num_hiddens)
        return output, state                                # state.shape = (num_layers, batch_size, num_hiddens)
    
class Seq2SeqDecoder(Decoder):
    """Decoder for sequence to sequence model."""
    def __init__(
        self, 
        vocab_size: int,
        embed_size: int,
        num_hiddens: int,
        num_layers: int,
        device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size 
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens 
        self.num_layers = num_layers
        self.device = device 
        
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)
        self.rnn = nn.GRU(input_size = embed_size + num_hiddens, hidden_size = num_hiddens, num_layers = num_layers)
        self.dense = nn.LazyLinear(vocab_size)
        
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs
        
    def forward(self, X: torch.Tensor, state):
        embeds = self.embedding(X.t().type(torch.int64))
        enc_output, hidden_state = state 
        
        context = enc_output[-1]
        context = context.repeat(embeds.shape[0], 1, 1)
        embeds_and_context = torch.cat((embeds, context), -1)
        outputs, hidden_state = self.rnn(embeds_and_context, hidden_state)
        
        outputs = self.dense(outputs).swapaxes(0, 1)
        
        return outputs, [enc_output, hidden_state]
    
class Seq2Seq(EncoderDecoder):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        tgt_pad: int,
        lr: float = 1e-3
    ):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_pad = tgt_pad
        self.lr = lr  
        self.criterion = F.cross_entropy
    
    def __shared_step(self, batch, batch_idx):
        self.logits = self(*batch)
        return self.criterion(self.logits.swapdims(1, 2), batch[-1])
    
    def training_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx)    
        
    def configure_optimizers(self):
        return torch.optim.Adam(params = self.parameters(), lr = self.lr)
    
    def loss(self, Y: torch.Tensor, Y_hat: torch.Tensor):
        mask = (Y.reshape(-1) != self.tgt_pad).type(torch.float32).requires_grad_()
        return (1 * mask).sum() / mask.sum()