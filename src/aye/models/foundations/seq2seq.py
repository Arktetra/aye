from aye.models.foundations.encoder_decoder import Encoder, Decoder, EncoderDecoder
from aye.models.init import init_seq2seq_weights

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
        dropout: float = 0.1,
        device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size 
        self.embed_size = embed_size 
        self.num_hiddens = num_hiddens
        self.device = device
        
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)
        self.rnn = nn.GRU(
            input_size = embed_size, hidden_size = num_hiddens, 
            num_layers = num_layers, dropout = dropout
        )
        
        self.apply(init_seq2seq_weights)
        
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
        dropout: float = 0.1,
        device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size 
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens 
        self.num_layers = num_layers
        self.device = device 
        
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)
        self.rnn = nn.GRU(
            input_size = embed_size + num_hiddens, hidden_size = num_hiddens, 
            num_layers = num_layers, dropout = dropout
        )
        self.dense = nn.LazyLinear(vocab_size)
        
        self.apply(init_seq2seq_weights)
        
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs
        
    def forward(self, X: torch.Tensor, state):
        embeds = self.embedding(X.t().type(torch.int32))
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
        return self.loss(self.logits.swapdims(1, 2), batch[-1])
    
    def training_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx)    
        
    def configure_optimizers(self):
        return torch.optim.Adam(params = self.parameters(), lr = self.lr)
    
    def loss(self, Y_hat: torch.Tensor, Y: torch.Tensor):
        """Masked cross entropy loss."""
        l = self.criterion(Y_hat, Y, reduction = "none")
        mask = (Y != self.tgt_pad).type(torch.float32)
        return (l * mask).sum() / mask.sum()
    
    def predict_step(self, batch, device, num_steps, save_attention_weights = False):
        batch = [b.to(device) for b in batch]
        src, tgt, src_valid_len, _ = batch 
        
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
                
        outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []
    
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(Y.argmax(dim = 2))
            
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
                
        return torch.cat(outputs[:], dim = 1), attention_weights
            
        