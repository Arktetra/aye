from aye.callbacks import Callback, MetricsCallback
from fastprogress import master_bar

from IPython import display

import aye
import collections
import matplotlib.pyplot as plt

def mean(x):
    return sum(x) / len(x)

class ProgressBoard(Callback):
    """
    Callback for displaying progress.
    
    Note: This callback should be placed after metric callback.
    """
    
    order = MetricsCallback.order + 1
    
    def __init__(
        self,
        xlabel = None,
        ylabel = None,
        xlim = None,
        ylim = None,
        xscale = "linear",
        yscale = "linear",
        ls = ["-", "--", "-.", ":"],
        colors = ["C0", "C1", "C2", "C2"],
        fig = None,
        axes = None,
        figsize = (8, 6),
        display = True
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.ls = ls
        self.colors = colors
        self.fig = fig
        self.axes = axes
        self.figsize = figsize
        self.display = display 
        
    def draw(self, x, y, label, every_n = 1):
        Point = collections.namedtuple("Point", ["x", "y"])
        
        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
            
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
            
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        
        if len(points) != every_n:
            return 
        
        line.append(Point(
            mean([p.x for p in points]),
            mean([p.y for p in points])
        ))
        
        points.clear()
        
        if not self.display:
            return 
        
        if self.fig is None:
            self.fig = plt.figure(figsize = self.figsize)
        
        plt_lines, labels = [], []
        
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                      linestyle = ls, color = color)[0])
            labels.append(k)
            
        axes = self.axes if self.axes else plt.gca()
        if self.xlim:
            axes.set_xlim(self.xlim)
        if self.ylim:
            axes.set_ylim(self.ylim)
        if not self.xlabel:
            self.xlabel = self.x 
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait = True)
        
    def before_fit(self, learner: "aye.Learner"):
        self.idx = 0
        
    def after_batch(self, learner: "aye.Learner"):
        if learner.training:
            self.draw(self.idx, learner.metrics.train_loss.compute(), "train_loss", every_n = 1)
            
        self.idx += 1
        
        # print(self.idx)