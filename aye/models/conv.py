import torch.nn as nn 

def conv(in_channels, out_channels, kernel_size = 3, stride = 2, act = None):
    """
    Return a convolution layer followed by an activation layer.
    """
    conv_layer = nn.Conv2d(
        in_channels = in_channels, out_channels = out_channels,
        kernel_size = kernel_size, stride = stride, padding = kernel_size // 2
    )
    
    if act: 
        return nn.Sequential(conv_layer, act)
    else:
        return conv_layer