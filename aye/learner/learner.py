from aye import AyeModule
from aye.callbacks import Callback, with_callbacks, run_callbacks, MetricsCallback
from torch.utils.data import DataLoader
from typing import Optional, Sequence

import torch

TRAIN_DATALOADER = DataLoader
VAL_DATALOADER = DataLoader

class Learner:
    def __init__(self, accelerator: str = None, callbacks: Sequence[Callback] = [MetricsCallback()]) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.callbacks = callbacks
        self.log_dict = {}
        
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
                        
        self.loss = loss / len(batch)
            
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
        epochs: Optional[int] = 5,
        lr: Optional[float] = 1e-3
    ) -> None:    
        if self.accelerator == "cuda":
            model.to("cuda")
            
        self.epochs = epochs
        self.optimizer = model.configure_optimizers(lr)
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.fit_epoch(model, train_dataloader, val_dataloader)
                                
    def callback(self, method_name):
        run_callbacks(self.callbacks, method_name, self)