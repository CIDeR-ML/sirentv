import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from functools import partial
import glob
import importlib
from abc import ABC, abstractmethod

wandb = None
import matplotlib.pyplot as plt
import numpy as np

try:
    from fast_histogram import histogram1d
except ImportError:
    print('fast_histogram not installed, using numpy')
    histogram1d = lambda *args,**kwargs: np.histogram(*args,**kwargs)[0]
    
    
    
def hist(a, bins=None, **kwargs):
    log = False
    if bins[1]-bins[0] != bins[2]-bins[1]:
        log = True
        
    data = np.log10(a+a[a!=0].min()/10) if log else a
        
    range = [bins.min(),bins.max()]
    if log:
        range = list(map(np.log10, range))
    bins  = len(bins)
    
    counts = histogram1d(data, bins=bins, range=range).astype(float)
    edges = np.linspace(*range, bins+1)
    
    if kwargs.pop('density', False):
        counts /= np.diff(edges) * counts.sum()

    if log:
        edges = 10**edges
    centers = (edges[1:]+edges[:-1])/2
        
    plt.gca().plot(centers, counts, drawstyle='steps-mid', **kwargs)
        
    

class WeightedL2Loss(nn.Module):
    """
    A simple loss module that implements a weighted MSE loss
    """

    def __init__(self, reduce_method=torch.mean):
        super().__init__()
        self.reduce = reduce_method

    def forward(self, pred, target, weight=1.0):
        loss = weight * (pred - target) ** 2
        return self.reduce(loss)

class L2Loss(nn.Module):
    """
    A simple loss module that implements a regular MSE loss
    """

    def __init__(self, reduce_method=torch.mean):
        super().__init__()
        self.reduce = reduce_method

    def forward(self, pred, target, weight=1.0):
        loss = (pred - target) ** 2
        return self.reduce(loss)

    
class WeightedCosineDissimilarity(nn.Module):
    """
    A simple loss module that implements a weighted cosine dissimilarity loss

    To be used when training on the *shape* of an output (e.g., PMT waveform), and not
    the explicit values.
    """

    def __init__(self, reduce_method=torch.mean):
        super().__init__()
        self.reduce = reduce_method
    
    def forward(self, pred, target, weight=1.0):
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        # normalize_weights = F.normalize(weight, p=2, dim=-1) if hasattr(weight, 'norm') else weight
        loss = (weight * (1 - (pred_norm * target_norm))).mean(dim=-1)
        return self.reduce(loss)
    

class LNRegularization(nn.Module):
    def __init__(self, weight_decay, p=2):
        super().__init__()
        self.weight_decay = weight_decay
        self.p = p
    
    def forward(self, net):
        return self.weight_decay * torch.norm(net.parameters(), p=self.p)

class L2Regularization(LNRegularization):
    def __init__(self, weight_decay):
        super().__init__(weight_decay, p=2)

class L1Regularization(LNRegularization):
    def __init__(self, weight_decay):
        super().__init__(weight_decay, p=1)


class UncertainMSE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weights, log_sigma):
        sigma = torch.exp(log_sigma)
        loss = torch.mean(weights * (((target - pred) ** 2) / (sigma ** 2) + log_sigma))
        return loss


class Logger(ABC):
    @abstractmethod
    def record(self, keys: list, vals: list):
        pass

    @abstractmethod
    def step(self, iteration, label=None, pred=None):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def write(self):
        pass




class WandbLogger(Logger):
    """
    Logger class to log training progress using Weights & Biases (wandb).
    """

    def __init__(self, cfg):
        """
        Constructor

        Parameters
        ----------
        cfg : dict
            A collection of configuration parameters. 'project' and 'name' specify
            the wandb project and run name respectively.
        """
        global wandb
        if wandb is None:
            import wandb
        wandb.require("core")

        log_cfg = cfg.get("logger", dict())
        self.project = log_cfg.get("project", "default-project")
        self.name = log_cfg.get("name", None)
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 1)
        self._logdir = self.make_logdir(log_cfg.get("dir_name", "logs"))
        self._logfile = os.path.join(self._logdir, cfg.get("file_name", "log.csv"))

        # Initialize wandb
        wandb.init(project=self.project, name=self.name, config=cfg)

        print(f"[WandbLogger] Initialized wandb project: {self.project}")

        self._dict = {}
        self._analysis_dict = {}

        for key, kwargs in log_cfg.get("analysis", dict()).items():
            print("[WandbLogger] adding analysis function:", key)
            self._analysis_dict[key] = partial(
                getattr(importlib.import_module("slar.analysis"), key), **kwargs
            )

    def record(self, keys: list, vals: list):
        """
        Function to register key-value pair to be logged

        Parameters
        ----------
        keys : list
            A list of parameter names to be logged.

        vals : list
            A list of parameter values to be logged.
        """
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def step(self, iteration, label=None, pred=None):
        """
        Function to take an iteration step during training/inference. If this step is
        subject for logging, this function logs the parameters registered through the record function.

        Parameters
        ----------
        iteration : int
            The current iteration for the step. If it's not modulo the specified steps to
            record a log, the function does nothing.

        label : torch.Tensor
            The target values (labels) for the model run for training/inference.

        pred : torch.Tensor
            The predicted values from the model run for training/inference.
        """
        if not iteration % self._log_every_nsteps == 0:
            return

        if None not in (label, pred):
            for key, f in self._analysis_dict.items():
                self.record([key], [f(label, pred)])
        self.write()

    def close(self):
        """
        Finish the wandb run.
        """
        wandb.finish()

    def write(self):
        """
        Log the key-value pairs provided through the record function to wandb.
        """
        wandb.log(self._dict)

    def save(self, path):
        """
        Save the wandb run to a file.
        """
        wandb.save(path)

    @property
    def logfile(self):
        return self._logfile

    @property
    def logdir(self):
        return self._logdir

    def make_logdir(self, dir_name):
        """
        Create a log directory

        Parameters
        ----------
        dir_name : str
            The directory name for a log file. There will be a sub-directory named version-XX where XX is
            the lowest integer such that a subdirectory does not yet exist.

        Returns
        -------
        str
            The created log directory path.
        """
        versions = [
            int(d.split("-")[-1])
            for d in glob.glob(os.path.join(dir_name, "version-[0-9]*"))
        ]
        ver = 0
        if len(versions):
            ver = max(versions) + 1
        logdir = os.path.join(dir_name, "version-%02d" % ver)
        os.makedirs(logdir)

        return logdir
    
    def watch_grad(self, net):
        wandb.watch(net, log="all", log_freq=100)


class CSVLogger(Logger):
    """
    Logger class to store training progress in a CSV file.
    """

    def __init__(self, cfg):
        """
        Constructor

        Parameters
        ----------
        cfg : dict
            A collection of configuration parameters. `dir_name` and `file_name` specify
            the output log file location. `analysis` specifies analysis function(s) to be
            created from the analysis module and run during the training.
        """

        log_cfg = cfg.get("logger", dict())
        self._logdir = self.make_logdir(log_cfg.get("dir_name", "logs"))
        self._logfile = os.path.join(self._logdir, cfg.get("file_name", "log.csv"))
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 1)

        print("[CSVLogger] output log directory:", self._logdir)
        print(f"[CSVLogger] recording a log every {self._log_every_nsteps} steps")
        self._fout = None
        self._str = None
        self._dict = {}

        self._analysis_dict = {}

        for key, kwargs in log_cfg.get("analysis", dict()).items():
            print("[CSVLogger] adding analysis function:", key)
            self._analysis_dict[key] = partial(
                getattr(importlib.import_module("slar.analysis"), key), **kwargs
            )

    @property
    def logfile(self):
        return self._logfile

    @property
    def logdir(self):
        return self._logdir

    def make_logdir(self, dir_name):
        """
        Create a log directory

        Parameters
        ----------
        dir_name : str
            The directory name for a log file. There will be a sub-directory named version-XX where XX is
            the lowest integer such that a subdirectory does not yet exist.

        Returns
        -------
        str
            The created log directory path.
        """
        versions = [
            int(d.split("-")[-1])
            for d in glob.glob(os.path.join(dir_name, "version-[0-9]*"))
        ]
        ver = 0
        if len(versions):
            ver = max(versions) + 1
        logdir = os.path.join(dir_name, "version-%02d" % ver)
        os.makedirs(logdir)

        return logdir

    def record(self, keys: list, vals: list):
        """
        Function to register key-value pair to be stored

        Parameters
        ----------
        keys : list
            A list of parameter names to be stored in a log file.

        vals : list
            A list of parameter values to be stored in a log file.
        """
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def step(self, iteration, label=None, pred=None):
        """
        Function to take a iteration step during training/inference. If this step is
        subject for logging, this function 1) runs analysis methods and 2) write the
        parameters registered through the record function to an output log file.

        Parameters
        ----------
        iteration : int
            The current iteration for the step. If it's not modulo the specified steps to
            record a log, the function does nothing.

        label : torch.Tensor
            The target values (labels) for the model run for training/inference.

        pred : torch.Tensor
            The predicted values from the model run for training/inference.


        """
        if not iteration % self._log_every_nsteps == 0:
            return

        if not None in (label, pred):
            for key, f in self._analysis_dict.items():
                self.record([key], [f(label, pred)])
        self.write()

    def write(self):
        """
        Function to write the key-value pairs provided through the record function
        to an output log file.
        """
        if self._str is None:
            self._fout = open(self._logfile, "w")
            self._str = ""
            for i, key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(",")
                    self._str += ","
                self._fout.write(key)
                self._str += "{:f}"
            self._fout.write("\n")
            self._str += "\n"
        self._fout.write(self._str.format(*(self._dict.values())))
        self.flush()

    def flush(self):
        """
        Flush the output file stream.
        """
        if self._fout:
            self._fout.flush()

    def close(self):
        """
        Close the output file.
        """
        if self._str is not None:
            self._fout.close()

