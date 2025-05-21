from aye import AyeModule

import torch.nn as nn

class Encoder(nn.Module):
    """Base class for encoder."""
    def __init__(self):
        super().__init__()
        
    def forward(self, X, *args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    """Base class for decoder."""
    def __init__(self):
        super().__init__()
        
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError 
    
    def forward(self, X, state):
        raise NotImplementedError
    
class EncoderDecoder(AyeModule):
    """Base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        
    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_X, dec_state)[0]
    
    
    