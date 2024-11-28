import torch.nn as nn 

def init_cnn_weights(module, leaky = 0.):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight)
        
def init_seq2seq_weights(module):
    """Initialize weights for sequence-to-sequence model."""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])