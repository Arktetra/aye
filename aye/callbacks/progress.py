from aye.callbacks import Callback

import aye

class ProgressCallback(Callback):
    """Callback for displaying progress."""
    def after_epoch(self, learner):
        learner.log()