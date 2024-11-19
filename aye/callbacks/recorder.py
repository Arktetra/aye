from aye.callbacks import Callback

import aye
import matplotlib.pyplot as plt

class RecorderCallback(Callback):
    """Callback for recording metrics per batch."""
    order = 1
    
    def __init__(self):
        super().__init__()
        self.losses = {
            "train_loss": [],
            "val_loss": []
        }
        
    def after_batch(self, learner: "aye.Learner"):
        if learner.training:
            self.losses["train_loss"].append(learner.metrics.train_loss.compute())
        else:
            self.losses["val_loss"].append(learner.metrics.val_loss.compute())
            
    def plot(self):
        plt.plot(self.losses["train_loss"], label = "train_loss")
        plt.plot(self.losses["val_loss"], label = "val_loss")
        plt.legend()
        plt.show()