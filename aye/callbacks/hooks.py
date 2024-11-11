from aye.utils import to_cpu, show_image, get_grid
from aye.callbacks import Callback
from collections.abc import Callable
from functools import partial
from typing import Optional

import aye
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class Hook():
    def __init__(self, module, fn):
        self.hook = module.register_forward_hook(partial(fn, self))
        
    def remove(self):
        self.hook.remove()
        
    def __del__(self):
        self.remove()
        
class Hooks(list):
    def __init__(self, modules, fn):
        super().__init__([Hook(module, fn) for module in modules])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for hook in self:
            hook.remove()
            
class HooksCallback(Callback):
    def __init__(self, hookfunc: Callable, modules, module_filter):
        super().__init__()
        self.hookfunc = hookfunc
        self.modules = modules
        self.module_filter = module_filter

    def before_fit(self, learner: "aye.Learner"):
        modules = []
        
        for mod in self.modules:
            if isinstance(mod, self.module_filter):
                modules += [mod]
        
        self.hooks = Hooks(modules, partial(self._hookfunc, learner))

    def _hookfunc(self, learner, *args, **kwargs):
        if learner.training:
            self.hookfunc(*args, **kwargs)

    def after_fit(self, learner):
        self.hooks.remove()

    def __iter__(self):
        return iter(self.hooks)

    def __len__(self):
        return len(self.hooks)
    
def append_stats(hook, module, ip, op):
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])

    acts = to_cpu(op)
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(bins = 40, min = 0, max = 10))
    
class ActivationStats(HooksCallback):
    """
    Callback for computing activation stats of the activation layer.

    # Args:
        `modules (nn.Module):` the modules making a composite module.
        `module_filter (nn.Module, optional):` module which represents activation 
        layers whose stats is to be computed. Defaults to None.
        
    # Example:
    
    Assuming we have a model which is an instance of `AyeModule`,
    
    >>> from aye.callbacks import MetricsCallback, LRFinderCallback, ActivationStats, EarlyStopping
    >>> from aye import Learner

    >>> from torcheval.metrics import MulticlassAccuracy
    
    >>> # Code for creating the model here
    
    >>> act_stats = ActivationStats(modules = list(model.modules())[1:], module_filter = nn.ReLU)
    >>> callbacks = [act_stats, MetricsCallback(accuracy = MulticlassAccuracy())]
    >>> learner = Learner(accelerator = "cuda", callbacks = callbacks)
    >>> learner.fit(model, train_dl, val_dl, lr = lr_finder.suggest())
    
    """
    
    def __init__(self, modules: nn.Module, module_filter: Optional[nn.Module] = None):
        super().__init__(hookfunc = append_stats, modules = modules, module_filter = module_filter)

    def color_dim(self, figsize = (11, 5)):
        fig, axes = get_grid(len(self), figsize = figsize)
        
        for ax, h in zip(axes.flat, self):
            show_image(self.get_hist(h), ax, origin = "lower")

    def dead_chart(self, figsize = (11, 5)):
        fig, axes = get_grid(len(self), figsize = figsize)
        
        for ax, h in zip(axes.flat, self):
            ax.plot(self.get_min(h))
            ax.set_ylim(0, 1)

    def plot_stats(self, figsize = (10, 4)):
        fig, axes = plt.subplots(1, 2, figsize = figsize)
        for h in self:
            for i in 0, 1:
                axes[i].plot(h.stats[i])
        axes[0].set_title("Means")
        axes[1].set_title("Stdevs")
        plt.legend(range(len(self)), bbox_to_anchor=(1.0, 1), loc='upper left')
        
    def get_hist(self, hook):
        return torch.stack(hook.stats[2]).t().float().log1p()
    
    def get_min(self, hook):
        hook_temp = torch.stack(hook.stats[2]).t().float()
        return hook_temp[0] / hook_temp.sum(0)
    
