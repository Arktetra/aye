import torch.nn as nn 

def init_cnn_weights(module, leaky = 0.):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight)