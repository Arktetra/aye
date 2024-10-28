"""The AyeModule - an nn.Module with additional features."""

from typing import Any
from typing_extensions import override

import torch
import torch.nn as nn

class AyeModule(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)
        
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("`training_step` must be implemented to be used with the Aye Learner.")
    
    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
        
    def test_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass 
    
    def predict_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        batch = kwargs.get("batch", args[0])
        return self(batch)
    
    def configure_optimizers(self):
        raise NotImplementedError("`configure_optimizer` must be implemented to be used with the Aye Learner.")
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()
    
    def optimizer_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad()
    
    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)