from aye import AyeModule
from torch.utils.data import DataLoader
from typing import Optional

import torch

TRAIN_DATALOADER = DataLoader
VAL_DATALOADER = DataLoader

class Learner:
    def __init__(self, accelerator: str = None, epochs = 5) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.epochs = epochs
        
    def fit(
        self,
        model: AyeModule,
        train_dataloader: Optional[TRAIN_DATALOADER] = None,
        val_dataloader: Optional[VAL_DATALOADER] = None,
    ) -> None:
        train_loss, val_loss = 0., 0.
        
        if self.accelerator == "cuda":
            model.to("cuda")
        
        optimizer = model.configure_optimizers()
        
        for epoch in range(self.epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                batch = batch[0].to("cuda"), batch[1].to("cuda")
                loss = model.training_step(batch, batch_idx)
                model.backward(loss)
                model.optimizer_step(optimizer)
                model.optimizer_zero_grad(optimizer)
                train_loss += loss / len(batch)
            train_loss /= len(train_dataloader)
            
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    batch = batch[0].to("cuda"), batch[1].to("cuda")
                    loss = model.validation_step(batch, batch_idx)
                    val_loss += loss / len(batch)
                val_loss /= len(val_dataloader)
                
            print(f"epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        