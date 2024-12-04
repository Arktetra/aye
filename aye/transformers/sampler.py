from aye.transformers.transformer import Transformer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast 

import numpy as np
import torch 
import torch.distributions as D

class TransformerSampler:
    def __init__(self, model: Transformer, tokenizer: GPT2TokenizerFast, device = None):
        self.model = model 
        self.cfg = model.config
        self.tokenizer = tokenizer
        self.device = device 
        
    @torch.inference_mode()
    def sample(self, prompt: str, max_tokens_generated = 100, verbose = False, **kwargs):
        """_summary_

        Args:
            prompt (str): input prompt.
            max_tokens_generated (int, optional): maximum number of tokens to generate. Defaults to 100.
            verbose (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_
        """
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, return_tensors = "pt").to(self.device)[0]
        for i in range(max_tokens_generated):
            logits = self.model(tokens.view(1, -1))     # (seq_len) -> (batch seq_len)
            logits = logits[-1, -1]
            next_token = torch.tensor(
                [self.sample_next_token(
                    tokens.squeeze(), 
                    logits, 
                    **kwargs)],
                device = self.device
            )
            tokens = torch.cat([tokens, next_token], dim = -1)
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break
            
        return self.tokenizer.decode(tokens)
    
    @torch.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        verbose = False
    ) -> list[tuple[float, torch.Tensor]]:
        """
        Sample text starting from the prompt. Sampling terminates at 
        max_tokens_generated, or when the model generates an end-of-sequence token.

        Args:
            prompt (str): input prompt.
            num_return_sequences (int): number of sequences to return. It should be less than num_beams.
            num_beams (int): number of beams to use.
            max_new_tokens (int): maximum number of new tokens to search for.
            no_repeat_ngram_size (int, optional): prevent ngram of size n from repeating. Defaults to 0.
            verbose (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            list[tuple[float, torch.Tensor]]: Autoregressively generated text, starting from the prompt.
        """
        raise NotImplementedError 
    
    @staticmethod
    def sample_next_token(
        input_ids: torch.Tensor,    
        logits: torch.Tensor,       
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = None
    ) -> int:
        """
        Samples the next token.

        Args:
            input_ids (torch.Tensor): token ids of shape(seq_len).
            logits (torch.Tensor): logits of shape(d_vocab).
            temperature (float, optional): scaling value. Defaults to 1.0.
            top_k (int, optional): number of logits to sample from. Defaults to 0.
            top_p (float, optional): cumulative probability value. Defaults to 0.0.
            frequency_penalty (float, optional): frequency penalty value. Defaults to 0.0.
            seed (int, optional): seed for determinism. Defaults to None.

        Returns:
            int: token id of the next token.
        """
        
        assert input_ids.ndim == 1, \
            "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, \
            "temperature must be non-negative"
        assert 0 <= top_p <= 1.0, \
            "top-p must be in [0, 1]"
        assert 0 <= top_k, \
            "top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), \
            "choose either top_p or top_k"
            
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
            
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
            
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        
        return TransformerSampler.sample_basic(logits)
    
    @staticmethod 
    def greedy_search(
        logits: torch.Tensor,       # shape: d_vocab
    ) -> int:
        """
        Returns the most likely token.

        Args:
            logits (torch.Tensor): logits of shape(d_vocab).

        Returns:
            int: token id of the next token.
        """
        return logits.argmax().item()
    
    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,       
        temperature: float
    ) -> torch.Tensor:              
        """
        Applies temperature scaling to the logits.

        Args:
            logits (torch.Tensor): logits from transformer of shape(d_vocab).
            temperature (float): value used for scaling the logits.

        Returns:
            torch.Tensor: logits of shape(d_vocab).
        """
        assert temperature > 0
        return logits / temperature
    
    @staticmethod
    def apply_frequency_penalty(
        input_ids: torch.Tensor,    
        logits: torch.Tensor,      
        freq_penalty: float         
    ) -> torch.Tensor:             
        """
        Applies a frequency penalty to the logits.

        Args:
            input_ids (torch.Tensor): token ids of shape(seq_len).
            logits (torch.Tensor): logits from transformer of shape(d_vocab).
            freq_penalty (float): frequency penalty that is applied to the logits.

        Returns:
            torch.Tensor: logits of shape(d_vocab).
        """
        (vocab_size, ) = logits.shape
        freq = torch.bincount(input_ids, minlength = vocab_size)
        return logits - freq_penalty * freq
    
    @staticmethod 
    def sample_basic(
        logits: torch.Tensor        # shape: d_vocab
    ) -> int:
        """
        Samples from the distribution defined by the logits.

        Args:
            logits (torch.Tensor): logits of shape (d_vocab).

        Returns:
            int: token id.
        """
        dist = D.categorical.Categorical(logits = logits)
        sampled_token = dist.sample().item()
        return sampled_token
    
    @staticmethod 
    def sample_top_k(
        logits: torch.Tensor,       # shape: d_vocab
        k: int                      
    ) -> int:
        """
        Samples from the top k most likely tokens.

        Args:
            logits (torch.Tensor): logits from transformer of shape (d_vocab).

        Returns:
            int: token id of the next token.
        """
        pass 
    
    @staticmethod 
    def sample_top_p(
        logits: torch.Tensor,       # shape: d_vocab
        top_p: float,
        min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p 
        cumulative probability.

        Args:
            logits (torch.Tensor): logits from transformer of shape (d_vocab).
            min_tokens_to_keep (int, optional): minimum number of tokens to keep. Defaults to 1.

        Returns:
            int: token id of the next token.
        """
        pass