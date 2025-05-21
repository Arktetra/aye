from aye.callbacks import Callback
from tqdm.auto import tqdm

import aye

class ProgressBar(Callback):
    def __init__(self):
        super().__init__()
        
    def before_epoch(self, learner: "aye.Learner"):
        self.t = tqdm(total = learner.num_batches)
        
    def after_batch(self, learner: "aye.Learner"):
        self.t.update(1)
        
    def after_epoch(self, learner: "aye.Learner"):
        self.t.close()