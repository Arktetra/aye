from .callback import Callback
from aye.exceptions import CancelFitException

import aye

class EarlyStopping(Callback):
    """
    Callback for enabling early stopping when the validation loss starts to increase.
    
    It is automatically used by `Learner` with its default setting if `max_epochs` to `Learner` is `None`.
    """
    def __init__(self, patience: int = 1, min_delta: int = 0):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        
    def early_stop(self, validation_loss: float):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                raise CancelFitException()
            
    def after_epoch(self, learner: "aye.Learner"):
        self.early_stop(learner.metrics.val_loss.compute())