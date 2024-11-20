from aye import AyeModule
from aye.callbacks import (
    Callback, 
    with_callbacks, 
    run_callbacks, 
    MetricsCallback, 
    EarlyStopping, 
    ModelCheckpoint, 
    ProgressBar
)
from aye.utils import has_instance
from torch.utils.data import DataLoader
from typing import Optional, Sequence, Union

import torch

TRAIN_DATALOADER = DataLoader
VAL_DATALOADER = DataLoader

class Learner:
    def __init__(
        self, 
        accelerator: Optional[str] = None, 
        max_epochs: Optional[int] = None, 
        callbacks: Sequence[Callback] = [],
        grad_clip_val: Optional[Union[int, float]] = None,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True
    ) -> None:
        super().__init__()
        self.accelerator = accelerator if accelerator is not None else "cpu"
        self.epochs = max_epochs if max_epochs is not None else 1000
        self.callbacks = callbacks
        self.grad_clip_val = grad_clip_val
        self.enable_checkpointing = enable_checkpointing
        self.log_dict = {}
        
        # Make sure this is placed before EarlyStopping callback
        if not has_instance(self.callbacks, MetricsCallback):
            self.callbacks += [MetricsCallback()]

        if enable_progress_bar and not has_instance(self.callbacks, ProgressBar):
            self.callbacks.append(ProgressBar())
        
        if max_epochs is None:
            self.callbacks += [EarlyStopping(patience = 1, min_delta = 0)]
        
        if self.enable_checkpointing and not has_instance(self.callbacks, ModelCheckpoint):
            self.callbacks += [ModelCheckpoint(dir_path = "./ckpt")]
        
    def log(self):
        print(self.log_dict)

    @with_callbacks("batch")
    def fit_one_batch(
        self, 
        model: AyeModule,  
    ):
        batch = self.batch[0].to(self.accelerator), self.batch[1].to(self.accelerator)
                    
        if self.training:
            loss = model.training_step(batch, self.batch_idx)
            model.backward(loss)
            
            if self.grad_clip_val:
                self.clip_gradients(self.grad_clip_val, model)
            
            model.optimizer_step(self.optimizer)
            model.optimizer_zero_grad(self.optimizer)
        else:
            loss = model.validation_step(batch, self.batch_idx)
            
        self.preds = model.logits
                        
        self.loss = loss
            
    @with_callbacks("epoch")
    def fit_epoch(
        self, 
        model: AyeModule, 
        train_dataloader: TRAIN_DATALOADER, 
        val_dataloader: VAL_DATALOADER, 
    ):
        self.training = True
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            self.batch_idx, self.batch = batch_idx, batch
            self.fit_one_batch(model)
            
        self.training = False
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                self.batch_idx, self.batch = batch_idx, batch
                self.fit_one_batch(model)

    @with_callbacks("fit")
    def fit(
        self,
        model: AyeModule,
        train_dataloader: Optional[TRAIN_DATALOADER] = None,
        val_dataloader: Optional[VAL_DATALOADER] = None,
    ) -> None:    
        model.to(self.accelerator)
        
        self.aye_module = model
        self.num_batches = len(train_dataloader) + len(val_dataloader)
        self.optimizer = model.configure_optimizers()
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.fit_epoch(model, train_dataloader, val_dataloader)
            
    def clip_gradients(self, grad_clip_value: Union[int, float], model: AyeModule):
        """Clips the gradient using norm."""
        params = [p for p in model.parameters()]
        norm = torch.sqrt(sum(p.grad ** 2 for p in params))
        
        if norm > grad_clip_value:
            for param in params:
                param.grad[:] *= grad_clip_value / norm 
                                
    def callback(self, method_name: str):
        run_callbacks(self.callbacks, method_name, self)
    