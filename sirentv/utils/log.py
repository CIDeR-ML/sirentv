import os
from functools import partial
import glob
import importlib
from abc import ABC, abstractmethod

wandb = None

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
        self.entity = log_cfg.get("entity", None)
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 1)
        self._logdir = self.make_logdir(log_cfg.get("dir_name", "logs"))
        self._logfile = os.path.join(self._logdir, cfg.get("file_name", "log.csv"))

        # Initialize wandb
        proj_cfg = dict(project=self.project, name=self.name)
        if self.entity:
            proj_cfg["entity"] = self.entity
        wandb.init(**proj_cfg, config=cfg)

        print(f"[WandbLogger] Initialized wandb project: {self.project}")

        self._dict = {}
        self._analysis_dict = {}
        for kwargs in log_cfg.get("analysis", []):
            func = kwargs.pop("func", "")
            suffix = kwargs.pop("suffix", "")
            if suffix:
                suffix = f"_{suffix}"
            print("[WandbLogger] adding analysis function:", func+suffix)

            self._analysis_dict[func + suffix] = partial(
                getattr(importlib.import_module("sirentv.analysis"), func), **kwargs
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

    def plot(self, iteration, inferred: dict):
        """
        Log a plot of x vs y at the given iteration.

        Parameters
        ----------
        iteration : int
            Training iteration or epoch.
        inferred : dict[torch.Tensor]
        kind : str
            Type of plot: "line" or "scatter" (default).
        """

        if inferred is None:
            return

        x = inferred['x_value'].detach().cpu().numpy()
        visibility = inferred['visibility'].detach().cpu().numpy()
        pdf = inferred['pdf'].detach().cpu().numpy()
        cdf = inferred['cdf'].detach().cpu().numpy()

        import matplotlib.pyplot as plt

        # --- Visibility plot ---
        fig_vis, ax_vis = plt.subplots()
        ax_vis.scatter(visibility[:, 0], visibility[:, 1], label="Target vs Pred")
        ax_vis.set_xlabel("Target Visibility")
        ax_vis.set_ylabel("Predicted Visibility")
        ax_vis.set_title("PMT Visibility")
        ax_vis.legend()

        # --- PDF plot ---
        fig_pdf, ax_pdf = plt.subplots()
        ax_pdf.plot(x, pdf[:, 0], label="Target", color="navy")
        ax_pdf.plot(x, pdf[:, 1], label="Predicted", color="darkorange")
        ax_pdf.set_xlabel("Time (ns)")
        ax_pdf.set_ylabel("Value")
        ax_pdf.set_title("Waveform PDF")
        ax_pdf.legend()

        # --- CDF plot ---
        fig_cdf, ax_cdf = plt.subplots()
        ax_cdf.plot(x, cdf[:, 0], label="Target", color="navy")
        ax_cdf.plot(x, cdf[:, 1], label="Predicted", color="darkorange")
        if 't0' in inferred.keys():
            t0s = inferred['t0'].detach().cpu().numpy()
            ax_cdf.axvline(t0s[0], linestyle='--', label="Target T0", color="navy", alpha=0.6)
            ax_cdf.axvline(t0s[1], linestyle='--', label="Predicted T0", color="darkorange", alpha=0.6)
        ax_cdf.set_xlabel("Time (ns)")
        ax_cdf.set_ylabel("Value")
        ax_cdf.set_title("Waveform CDF")
        ax_cdf.legend()

        # log both figures in a single step
        wandb.log({
            "PMT Visibility": wandb.Image(fig_vis),
            "PDF": wandb.Image(fig_pdf),
            "CDF": wandb.Image(fig_cdf),
        }, step=iteration)

        plt.close(fig_vis)
        plt.close(fig_pdf)
        plt.close(fig_cdf)


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
            suffix = kwargs.pop("suffix", "")
            if suffix:
                suffix = f"_{suffix}"
            self._analysis_dict[key + suffix] = partial(
                getattr(importlib.import_module("sirentv.analysis"), key), **kwargs
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

