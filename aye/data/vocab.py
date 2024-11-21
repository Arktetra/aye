from typing import List, Union, Tuple

import collections
import torch

class Vocab:
    def __init__(self, tokens: List[str] = [], min_freq: int = 0, reserved_tokens: List[str] = []):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
            
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key = lambda x: x[1], reverse = True)
        
        self.idx_to_token = list(sorted(set(["<unk>"] + reserved_tokens + 
                                            [token for token, freq in self.token_freqs 
                                             if freq > min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens: Union[str, List[str], Tuple[str]]):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices: Union[int, List[int]]):
        if (isinstance(indices, torch.Tensor) and indices.ndim == 2) or isinstance(indices[0], list):
            tokens = []
            tokens.append([self.to_tokens(idxs) for idxs in indices])
            return tokens
        elif hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(idx)] for idx in indices]
        return self.idx_to_token[indices]
    
    @property
    def unk(self):
        return self.token_to_idx["<unk>"]