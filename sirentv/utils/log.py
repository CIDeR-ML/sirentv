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

    def __init__(self, cfg, rank=0, world_size=1, is_distributed=False):
        """
        Constructor

        Parameters
        ----------
        cfg : dict
            A collection of configuration parameters. 'project' and 'name' specify
            the wandb project and run name respectively.
        rank : int
            Global rank of this process
        world_size : int
            Total number of processes
        is_distributed : bool
            Whether running in distributed mode
        """
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed

        log_cfg = cfg.get("logger", dict())
        self.project = log_cfg.get("project", "default-project")
        self.name = log_cfg.get("name", None)
        self.entity = log_cfg.get("entity", None)
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 1)

        if self.rank == 0:
            self._logdir = self.make_logdir(log_cfg.get("dir_name", "logs"))
            self._logfile = os.path.join(self._logdir, cfg.get("file_name", "log.csv"))
        else:
            self._logdir = None
            self._logfile = None

        if self.rank == 0:
            global wandb
            if wandb is None:
                import wandb
            wandb.require("core")

            # Initialize wandb
            proj_cfg = dict(
                project=self.project,
                name=self.name,
                config={**cfg,
                        "world_size": world_size,
                        "distributed": is_distributed}
            )
            if self.entity:
                proj_cfg["entity"] = self.entity
            wandb.init(**proj_cfg, settings=wandb.Settings(start_method="fork"))
            self.wandb = wandb
            print(f"[WandbLogger] Initialized wandb project: {self.project}")
        else:
            self.wandb = None

        self._dict = {}
        self._analysis_dict = {}

        if self.rank == 0:
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
        Function to register key-value pair to be logged.
        Only rank 0 actually records.

        Parameters
        ----------
        keys : list
            A list of parameter names to be logged.

        vals : list
            A list of parameter values to be logged.
        """
        if self.rank == 0:
            for i, key in enumerate(keys):
                self._dict[key] = vals[i]

    def step(self, iteration, label=None, pred=None):
        """
        Function to take an iteration step during training/inference. If this step is
        subject for logging, this function logs the parameters registered through the record function.
        Only rank 0 actually records.

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

        if self.rank != 0:
            return

        if not iteration % self._log_every_nsteps == 0:
            return

        if None not in (label, pred):
            for key, f in self._analysis_dict.items():
                self.record(["analysis/"+key], [f(label, pred)])
        self.write()

    def plot(self, iteration, inferred: dict):
        """
        Log a plot of x vs y at the given iteration.
        Only rank 0 actually records.

        Parameters
        ----------
        iteration : int
            Training iteration or epozch.
        inferred : dict[torch.Tensor]
        kind : str
            Type of plot: "line" or "scatter" (default).
        """

        if self.rank != 0 or inferred is None or self.wandb is None:
            return

        x = inferred['x_value'].detach().cpu().numpy()
        visibility = inferred['visibility'].detach().cpu().numpy()
        pdf = inferred['pdf'].detach().cpu().numpy()
        cdf = inferred['cdf'].detach().cpu().numpy()

        import matplotlib.pyplot as plt

        # --- Visibility plot ---
        fig_vis, ax_vis = plt.subplots()
        ax_vis.scatter(visibility[:, 0], visibility[:, 1], label="Target vs Pred")
        max_val = max(visibility[:, 0].max(), visibility[:, 1].max())*1.1
        ax_vis.set_xlim(0, max_val)
        ax_vis.set_ylim(0, max_val)
        ax_vis.plot([0, 1], [0, 1], 'r--', alpha=0.8, label="y=x")
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
        # cdf plot with residual subplot below
        import numpy as np
        from matplotlib.gridspec import GridSpec

        fig_cdf = plt.figure(constrained_layout=True, figsize=(6, 6))
        gs = GridSpec(5, 1, figure=fig_cdf)
        ax_cdf = fig_cdf.add_subplot(gs[:4, 0])
        ax_res = fig_cdf.add_subplot(gs[4, 0], sharex=ax_cdf)

        ax_cdf.plot(x, cdf[:, 0], label="Target", color="navy")
        ax_cdf.plot(x, cdf[:, 1], label="Predicted", color="darkorange")
        if 't0' in inferred.keys() and inferred['t0'] is not None:
            t0s = inferred['t0'].detach().cpu().numpy()
            ax_cdf.axvline(t0s[0], linestyle='--', label="Target T0", color="navy", alpha=0.6)
            ax_cdf.axvline(t0s[1], linestyle='--', label="Predicted T0", color="darkorange", alpha=0.6)
        ax_cdf.set_xlabel("Time (ns)")
        ax_cdf.set_ylabel("Value")
        ax_cdf.set_title("Waveform CDF")
        ax_cdf.legend()

        # residual plot
        residual = cdf[:, 1] - cdf[:, 0]
        ax_res.plot(x, np.zeros_like(x), color="grey", linewidth=1, alpha=0.7)
        ax_res.plot(x, residual, color="red", linewidth=1)
        ax_res.set_ylabel("Residual")
        ax_res.set_xlabel("Time (ns)")
        ax_res.set_ylim(-np.max(np.abs(residual))*1.1, np.max(np.abs(residual))*1.1)
        ax_res.set_title("CDF Residual (Pred - Target)")
        ax_res.spines['top'].set_visible(False)
        ax_res.spines['right'].set_visible(False)

        # log both figures in a single step
        self.wandb.log({
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
        if self.wandb is not None:
            self.wandb.finish()

    def write(self):
        """
        Log the key-value pairs provided through the record function to wandb.
        """
        if self.wandb is not None:
            self.wandb.log(self._dict)
            self._dict = {}  # Clear dict after logging

    def save(self, path):
        """
        Save the wandb run to a file.
        """
        if self.wandb is not None:
            self.wandb.save(path)

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
        if self.wandb is not None:
            self.wandb.watch(net, log="all", log_freq=100)

    def log_aggregated_loss(self, iteration, loss_tensor):
        """
        Log loss aggregated across all ranks.
        All ranks must call this, but only rank 0 logs.

        Parameters
        ----------
        iteration : int
            Current iteration
        loss_tensor : torch.Tensor
            Loss tensor (must be on GPU for all_reduce)

        """
        if self.is_distributed:
            import torch.distributed as dist
            loss_avg = loss_tensor.detach().clone()
            dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
        else:
            loss_avg = loss_tensor.detach()
        if self.wandb is not None:
            self.wandb.log({
                "loss/avg_across_gpus": loss_avg.item(),
                "loss/rank_0": loss_tensor.item(),
            }, step=iteration)

    def log_per_rank_metrics(self, iteration, metrics_dict):
        """
        Gather and log per-rank metrics.
        All ranks must call this with their local metrics.

        Parameters
        ----------
        iteration : int
            Current iteration
        metrics_dict : dict
            Dict of {metric_name: tensor_value} for this rank
        """
        if not self.is_distributed:
            if self.wandb is not None:
                self.wandb.log(metrics_dict, step=iteration)
            return
        import torch
        import torch.distributed as dist
        log_dict = {}
        for metric_name, metric_value in metrics_dict.items():
            if not isinstance(metric_value, torch.Tensor):
                metric_value = torch.tensor(metric_value).cuda()

            gathered = [torch.zeros_like(metric_value) for _ in range(self.world_size)]
            dist.all_gather(gathered, metric_value)

            if self.wandb is not None:
                for rank_id, val in enumerate(gathered):
                    log_dict[f"{metric_name}/rank_{rank_id}"] = val.item()
                avg_val = torch.stack(gathered).mean()
                log_dict[f"{metric_name}/avg"] = avg_val.item()
        if self.wandb is not None:
            self.wandb.log(log_dict, step=iteration)

class CSVLogger(Logger):
    """
    Logger class to store training progress in a CSV file.
    """

    def __init__(self, cfg, rank=0, world_size=1, is_distributed=False):
        """
        Constructor

        Parameters
        ----------
        cfg : dict
            A collection of configuration parameters. `dir_name` and `file_name` specify
            the output log file location. `analysis` specifies analysis function(s) to be
            created from the analysis module and run during the training.
        rank : int
            Global rank of this process
        world_size : int
            Total number of processes
        is_distributed : bool
            Whether running in distributed mode
        """

        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed

        log_cfg = cfg.get("logger", dict())
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 1)
        if self.rank == 0:
            self._logdir = self.make_logdir(log_cfg.get("dir_name", "logs"))
            self._logfile = os.path.join(self._logdir, cfg.get("file_name", "log.csv"))
            print("[CSVLogger] output log directory:", self._logdir)
            print(f"[CSVLogger] recording a log every {self._log_every_nsteps} steps")
        else:
            self._logdir = None
            self._logfile = None

        self._fout = None
        self._str = None
        self._dict = {}
        self._analysis_dict = {}

        if self.rank == 0:
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
        if self.rank == 0:
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
        if self.rank != 0:
            return

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
        if self.rank != 0:
            return
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

