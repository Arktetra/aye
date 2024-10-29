from aye.callbacks import Callback
from aye.exceptions import CancelEpochException, CancelFitException
from aye.utils import to_cpu

import math
import matplotlib.pyplot as plt
import torch

class LRFinderCallback(Callback):
    def __init__(self, lr_multiplier = 1.3):
        self.lr_multiplier = lr_multiplier

    def before_fit(self, learner) -> None:
        self.lrs, self.losses = [], []
        self.min = math.inf

    def after_batch(self, learner) -> None:
        if not learner.training:
            raise CancelEpochException()
        self.lrs.append(learner.optimizer.param_groups[0]["lr"])
        loss = to_cpu(learner.loss)
        self.losses.append(loss)

        if loss < self.min:
            self.min = loss

        if loss > self.min * 3:
            raise CancelFitException()

        for g in learner.optimizer.param_groups:
            g["lr"] *= self.lr_multiplier

    def plot(self, grads: bool = False) -> None:
        plt.plot(self.lrs, self.losses, label = "loss")
        if grads:
            loss_grads = torch.gradient(torch.tensor(self.losses))[0]
            steepest_slope_idx = torch.argmax(-loss_grads)
            plt.plot(self.lrs, loss_grads, label = "loss gradient")
            plt.axvline(self.lrs[steepest_slope_idx], color = "green", ls = ":", label = "steepest slope")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.xscale("log")

    def suggest(self) -> float:
        loss_grads = torch.gradient(torch.tensor(self.losses))[0]
        steepest_slope_idx = torch.argmax(-loss_grads)
        return self.lrs[steepest_slope_idx]