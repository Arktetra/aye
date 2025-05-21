from .callback import Callback
from pathlib import Path
from typing import Union

import aye
import torch

class ModelCheckpoint(Callback):
    
    def __init__(self, dir_path: Union[Path, str]):
        super().__init__()
        self.dir_path = dir_path
        self.best_val_loss = float("inf")
        self.best_model_path = None
        
    
    def after_epoch(self, learner: "aye.Learner"):
        if not learner.enable_checkpointing:
            return
        
        val_loss = learner.metrics.val_loss.compute()
        
        filename = f"{type(learner.aye_module).__name__}-{learner.epoch}-{val_loss:.4f}"
        
        if not isinstance(self.dir_path, Path):
            self.dir_path = Path(self.dir_path)
            
        self.dir_path.mkdir(parents = True, exist_ok = True)
        
        ckpt_path = self.dir_path / filename 
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_path = ckpt_path
        
        torch.save(learner.aye_module.state_dict(), ckpt_path)
        