from jaxtyping import Float, Int

import einops
import torch 
import torch.nn as nn

class Embedding(nn.Module):
    """Implementation of Embedding."""
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.W_E = nn.Parameter(torch.empty((config.d_vocab, config.d_model)))
        nn.init.normal_(self.W_E, std = self.config.init_range)
        
    def forward(
        self, 
        tokens: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        return self.W_E[tokens]
    
class PosEmbedding(nn.Module):
    """Implementation of Positional Embedding."""
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.W_pos = nn.Parameter(torch.empty((config.n_ctx, config.d_model)))
        nn.init.normal_(self.W_pos, std = config.init_range)
        
    def forward(
        self, 
        tokens: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Forward method for positional embedding layer.

        Args:
            tokens (torch.Tensor): batch of tokens of shape: (batch, position)

        Returns:
            torch.Tensor: position embedded tensors of shape: (batch, position, d_model)
        """
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch = batch)
    
class Unembedding(nn.Module):
    """Implementation of Unembedding."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_U = nn.Parameter(torch.empty((config.d_model, config.d_vocab)))
        self.b_U = nn.Parameter(torch.zeros((config.d_vocab), requires_grad = False))
        nn.init.normal_(self.W_U, std = config.init_range)
        
    def forward(
        self, 
        normalized_resid_final: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Int[torch.Tensor, "batch seq_len d_vocab"]:
        return einops.einsum(
            normalized_resid_final, self.W_U,
            "batch seq_len d_model, d_model vocab_size -> batch seq_len vocab_size"
        ) + self.b_U