from aye import AyeModule
from aye.callbacks import Callback, with_callbacks, run_callbacks, MetricsCallback, EarlyStopping, ModelCheckpoint
from aye.utils import has_instance
from torch.utils.data import DataLoader
from typing import Optional, Sequence

import torch

TRAIN_DATALOADER = DataLoader
VAL_DATALOADER = DataLoader

class Learner:
    def __init__(
        self, 
        accelerator: str = None, 
        max_epochs = None, 
        callbacks: Sequence[Callback] = [MetricsCallback()],
        enable_checkpointing: bool = True
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.epochs = max_epochs if max_epochs is not None else 1000
        self.callbacks = callbacks
        self.enable_checkpointing = enable_checkpointing
        self.log_dict = {}
        
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
        batch = self.batch[0].to("cuda"), self.batch[1].to("cuda")
                    
        if self.training:
            loss = model.training_step(batch, self.batch_idx)
            model.backward(loss)
            model.optimizer_step(self.optimizer)
            model.optimizer_zero_grad(self.optimizer)
        else:
            loss = model.validation_step(batch, self.batch_idx)
            
        self.preds = model.logits
                        
        # self.loss = loss / len(batch)
        self.loss = loss
            
    @with_callbacks("epoch")
    def fit_epoch(
        self, 
        model: AyeModule, 
        train_dataloader: TRAIN_DATALOADER, 
        val_dataloader: VAL_DATALOADER, 
    ):
        self.training = True
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
        lr: Optional[float] = 1e-3
    ) -> None:    
        if self.accelerator == "cuda":
            model.to("cuda")
        
        self.aye_module = model
    
        self.optimizer = model.configure_optimizers(lr)
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.fit_epoch(model, train_dataloader, val_dataloader)
                                
    def callback(self, method_name: str):
        run_callbacks(self.callbacks, method_name, self)