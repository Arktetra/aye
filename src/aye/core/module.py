"""The AyeModule - an nn.Module with additional features."""

from pathlib import Path
from typing import Any, Union
from typing_extensions import override

import torch
import torch.nn as nn

class AyeModule(nn.Module):
    """
    Create AyeModule.
        
    Examples::
        
        >>> from aye import AyeModule
        >>> import torch.nn as n
        >>> import torch.nn.functional as F
        >>>
        >>> class MNISTClassifier(AyeModule):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>
        >>>         self.layer_1 = nn.Linear(28 * 28, 128)
        >>>         self.layer_2 = nn.Linear(128, 10)
        >>>         self.criterion = F.nll_loss
        >>>
        >>>     def forward(self, x):
        >>>         x = x.view(x.size(0), -1)
        >>>         x = self.layer_1(x)
        >>>         x = F.relu(x)
        >>>         x = self.layer_2(x)
        >>>         x = F.log_softmax(x, dim = 1)
        >>>         return x
        >>>     
        >>>     def _shared_step(self, batch, batch_idx):
        >>>         x, y = batch
        >>>         self.logits = self(x)
        >>>         return self.criterion(self.logits, y)
        >>>
        >>>     def training_step(self, batch, batch_idx):
        >>>         return self._shared_step(batch, batch_idx)
        >>>
        >>>     def validation_step(self, batch, batch_idx):
        >>>         return self._shared_step(batch, batch_idx)
        >>>
        >>>     def test_step(self, batch, batch_idx):
        >>>         return self._shared_step(batch, batch_idx)
        >>>
        >>>     def configure_optimizers(self):
        >>>         return torch.optim.Adam(params = self.parameters())
    """
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
        
    def load_ckpt(self, ckpt_path: Union[Path, str]):
        self.load_state_dict(torch.load(ckpt_path, weights_only = True))
        self.eval()
        
    def summary(self, x_shape):
        """
        Print the module summary.
        
        # Example
        
        Assuming that `model` is an instance of `AyeModule`:

        >>> model.summary(x_shape = (1, 1, 28, 28))
        """
        X = torch.randn(*x_shape)
        
        input_size = X.numel()     # input size is changed in the loops
        pass_size = 0
        
        total_params = 0
        train_params = 0

        layer_width = 20
        oshape_width = 35
        params_width = 20
        total_width = layer_width + oshape_width + params_width + 3

        print("-" * total_width)
        print(
            "layer (type)".rjust(layer_width, " "), 
            "output size".rjust(oshape_width, " "),
            "params".rjust(params_width, " ")
        )
        print("=" * total_width)
        
        layers = []
        list_modules(list(self.modules())[1], layers)

        for layer in layers:
            X = layer(X)
            pass_size += X.numel()
            num_params = sum(p.numel() for p in layer.parameters())
            train_params += sum(p.numel() for p in layer.parameters() if p.requires_grad)
            total_params += num_params

            print(
                layer.__class__.__name__.rjust(layer_width, " "), 
                f"{X.shape}".rjust(oshape_width, " "),
                f"{num_params}".rjust(params_width, " ")
            )
        print("=" * total_width)

        print("Total params: ", total_params)
        print("Trainable params: ", train_params)
        print("Non-trainable params: ", total_params - train_params)

        print("-" * total_width)
        
        if layers[0].weight.dtype == torch.float16:
            dtype_size = 2      # size of a single parameter
        elif layers[0].weight.dtype == torch.float32:
            dtype_size = 4
        elif layers[0].weight.dtype == torch.float64:
            dtype_size = 8
            
        input_size = dtype_size * input_size / 1024 ** 2        # change into MB
        pass_size = 2 * dtype_size * pass_size / 1024 ** 2      # multiplied by 2 because there are two passes: forward and backward
        params_size = dtype_size * total_params / 1024 ** 2
        total_size = input_size + pass_size + params_size
            
        print(f"Input size (MB): {input_size:.3f}")
        print(f"Forward/backward pass (MB): {pass_size:.3f}")
        print(f"Params size (MB):  {params_size:.3f}")
        print(f"Estimated Total Size (MB): {total_size:.3f}")
        
        print("-" * total_width)
        
def list_modules(module, mod_list):
    for mod in module:
        if isinstance(mod, nn.Sequential):
            list_modules(mod, mod_list)
        else:
            mod_list.append(mod)