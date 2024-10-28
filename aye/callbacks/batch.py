from aye.callbacks import Callback
from aye.exceptions import CancelEpochException

import aye

class SingleBatchCallback(Callback):
    """
    Callback for running `Learner.fit` on only one batch in each epoch.
    """
    order = 0
    
    def after_batch(self, learner):
        raise CancelEpochException()