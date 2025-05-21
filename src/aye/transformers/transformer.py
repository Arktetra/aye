from aye.models.foundations.attention import Attention
from aye.models.foundations.embedding import Embedding, PosEmbedding, Unembedding
from aye.models.foundations.normalization import LayerNorm
from aye.utils import gelu_new
from dataclasses import dataclass
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

import einops
import torch
import torch.nn as nn

@dataclass
class Config :
    d_model: int = 768
    debug: bool = True 
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    

class MLP(nn.Module):
    """Implementation of MLP for transformer."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_in = nn.Parameter(torch.empty((config.d_model, config.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((config.d_mlp, config.d_model)))
        self.b_in = nn.Parameter(torch.zeros((config.d_mlp)))
        self.b_out = nn.Parameter(torch.zeros((config.d_model)))
        nn.init.normal_(self.W_in, std = config.init_range)
        nn.init.normal_(self.W_out, std = config.init_range)
        
    def forward(
        self, 
        residual: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        residual = einops.einsum(
            residual, self.W_in,
            "batch seq_len d_model, d_model d_mlp -> batch seq_len d_mlp"
        ) + self.b_in
        residual = gelu_new(residual)
        residual = einops.einsum(
            residual, self.W_out,
            "batch seq_len d_mlp, d_mlp d_model -> batch seq_len d_model"
        ) + self.b_out
        return residual
    
class TransformerBlock(nn.Module):
    """Implmentation of Transformer block for transformer."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)
        
    def forward(
        self, 
        resid_pre: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post

class Transformer(nn.Module):
    """Implementation of Transformer."""
    def __init__(self, config: Config, from_pretrained = False, device = None):
        super().__init__()
        self.config = config
        self.embed = Embedding(config)
        self.pos_embed = PosEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = LayerNorm(config)
        self.unembed = Unembedding(config)
        
        if from_pretrained:
            reference_gpt2 = HookedTransformer.from_pretrained(
                "gpt2-small",
                fold_ln = False,
                center_unembed = False,
                center_writing_weights = False,
                device = device
            )
            self.load_state_dict(reference_gpt2.state_dict(), strict = False)
        
    def forward(
        self, 
        tokens: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        residual = self.ln_final(residual)
        return self.unembed(residual)