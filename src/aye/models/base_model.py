from aye import AyeModule
from typing import Any
from typing_extensions import override

import torch

class BaseModel(AyeModule):
    """
    Base class used to build other models. 
    """
    def __init__(self):
        super().__init__()
        
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        self.logits = self(x)
        return self.criterion(self.logits, y)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(params = self.parameters(), lr = self.lr)
    
    def apply_init(self, inputs, init = None):
        self.forward(inputs)
        if init is not None:
            self.net.apply(init)