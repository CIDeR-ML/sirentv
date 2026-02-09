import os
from functools import partial
import glob
import numpy as np
import torch
import torch.distributed as dist

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
            self._wandb_rundir = log_cfg.get("run_dir", "./")
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
            wandb.init(**proj_cfg, dir=self._wandb_rundir, settings=wandb.Settings(start_method="fork"))
            self.wandb = wandb
            print(f"[WandbLogger] Initialized wandb project: {self.project}")
        else:
            self.wandb = None

        self._dict = {}
        self._analysis_dict = {}

        for kwargs in log_cfg.get("analysis", []):
            func = kwargs.pop("func", "")
            suffix = kwargs.pop("suffix", "")
            if suffix:
                suffix = f"_{suffix}"
            if self.rank == 0:
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

    def commit(self, iteration):
        """Commit all pending logs to wandb"""
        if iteration % self._log_every_nsteps == 0 and self.wandb is not None:
            self.wandb.log({}, step=iteration, commit=True)

    def step(self, iteration, label=None, pred=None):
        """
        Function to take an iteration step during training/inference. If this step is
        subject for logging, this function logs the parameters registered through the record function.
        Per-rank analysis logging.

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
        if iteration == 0 or iteration % self._log_every_nsteps != 0:
            return

        if None not in (label, pred) and len(self._analysis_dict)>0:
            local_metrics = {}
            for key, f in self._analysis_dict.items():
                metric_value = f(label, pred)
                local_metrics[f"analysis/{key}"] = metric_value
                #self.record(["analysis/"+key], [f(label, pred)])

            self.log_per_rank_metrics(iteration, local_metrics)

        if self.rank == 0:
            self.write()

    def plot(self, iteration, inferred: dict):
        """
        Log a plot of x vs y at the given iteration.

        Parameters
        ----------
        iteration : int
            Training iteration or epozch.
        inferred : dict[torch.Tensor]
        kind : str
            Type of plot: "line" or "scatter" (default).
        """

        if inferred is None:
            return

        local_data = {
            'x_value': inferred['x_value'].detach().cpu(),
            'visibility': inferred['visibility'].detach().cpu(),
            'pdf': inferred['pdf'].detach().cpu(),
            'cdf': inferred['cdf'].detach().cpu(),
        }
        if 'position' in inferred and inferred['position'] is not None:
            local_data['position'] = inferred['position'].detach().cpu()
        if 't0' in inferred and inferred['t0'] is not None:
            local_data['t0'] = inferred['t0'].detach().cpu()

        if self.is_distributed:

            gathered_data = [None for _ in range(self.world_size)]
            if self.rank == 0:
                dist.gather_object(local_data, gathered_data, dst=0)
            else:
                dist.gather_object(local_data, None, dst=0)
                return
            all_data = gathered_data
        else:
            # Non-distributed: just use local data
            if self.rank != 0:
                return
            all_data = [local_data]

        if iteration % self._log_every_nsteps != 0 or self.wandb is None:
            return

        n_ranks = len(all_data)
        n_cols = min(2, n_ranks)  # Max 2 columns
        n_rows = (n_ranks + n_cols - 1) // n_cols

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # --- Visibility plot ---
        fig_vis, axes_vis = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        fig_vis.suptitle("PMT Visibility - All Ranks", fontsize=14, fontweight='bold')

        for rank_id, data in enumerate(all_data):
            row = rank_id // n_cols
            col = rank_id % n_cols
            ax = axes_vis[row, col]

            visibility = data['visibility'].numpy()
            ax.scatter(visibility[:, 0], visibility[:, 1], label="Target vs Pred", alpha=0.6)
            max_val = max(visibility[:, 0].max(), visibility[:, 1].max()) * 1.1
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label="y=x")
            ax.set_xlabel("Target Visibility")
            ax.set_ylabel("Predicted Visibility")

            subtitle = f"Rank {rank_id}"
            if data['position'] is not None:
                pos = data['position'].numpy()
                if pos.ndim > 1:
                    pos = pos[0]  # Take first position if batch
                subtitle += f"\nPos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            ax.set_title(subtitle)
            ax.legend()
            ax.grid(True, alpha=0.3)

        for rank_id in range(n_ranks, n_rows * n_cols):
            row = rank_id // n_cols
            col = rank_id % n_cols
            axes_vis[row, col].axis('off')

        plt.tight_layout()

        # --- PDF plot ---
        fig_pdf, axes_pdf = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

        for rank_id, data in enumerate(all_data):
            row = rank_id // n_cols
            col = rank_id % n_cols
            ax = axes_pdf[row, col]

            x = data['x_value'].numpy()
            pdf = data['pdf'].numpy()

            ax.plot(x, pdf[:, 0], label="Target", color="navy", linewidth=2)
            ax.plot(x, pdf[:, 1], label="Predicted", color="darkorange", linewidth=2)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Value")

            subtitle = f"Rank {rank_id}"
            if data['position'] is not None:
                pos = data['position'].numpy()
                if pos.ndim > 1:
                    pos = pos[0]
                subtitle += f"\nPos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            ax.set_title(subtitle)
            ax.legend()
            ax.grid(True, alpha=0.3)

        for rank_id in range(n_ranks, n_rows * n_cols):
            row = rank_id // n_cols
            col = rank_id % n_cols
            axes_pdf[row, col].axis('off')

        plt.tight_layout()

        # --- CDF plot ---
        # cdf plot with residual subplot below

        fig_cdf = plt.figure(figsize=(6 * n_cols, 6 * n_rows))

        for rank_id, data in enumerate(all_data):
            row = rank_id // n_cols
            col = rank_id % n_cols

            # Position in the overall grid
            gs = GridSpec(n_rows * 5, n_cols, figure=fig_cdf,
                          hspace=0.4, wspace=0.3)

            ax_cdf = fig_cdf.add_subplot(gs[row * 5:row * 5 + 4, col])
            ax_res = fig_cdf.add_subplot(gs[row * 5 + 4, col], sharex=ax_cdf)

            x = data['x_value'].numpy()
            cdf = data['cdf'].numpy()

            ax_cdf.plot(x, cdf[:, 0], label="Target", color="navy", linewidth=2)
            ax_cdf.plot(x, cdf[:, 1], label="Predicted", color="darkorange", linewidth=2)

            # Add T0 lines if available
            if data['t0'] is not None:
                t0s = data['t0'].numpy()
                ax_cdf.axvline(t0s[0], linestyle='--', label="Target T0", color="navy", alpha=0.6)
                ax_cdf.axvline(t0s[1], linestyle='--', label="Predicted T0", color="darkorange", alpha=0.6)

            ax_cdf.set_ylabel("Value")

            subtitle = f"Rank {rank_id}"
            if data['position'] is not None:
                pos = data['position'].numpy()
                if pos.ndim > 1:
                    pos = pos[0]
                subtitle += f" - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            ax_cdf.set_title(subtitle, fontsize=10)
            ax_cdf.legend(fontsize=8)
            ax_cdf.grid(True, alpha=0.3)
            ax_cdf.tick_params(labelbottom=False)

            # Residual plot
            residual = cdf[:, 1] - cdf[:, 0]
            ax_res.plot(x, np.zeros_like(x), color="grey", linewidth=1, alpha=0.7)
            ax_res.plot(x, residual, color="red", linewidth=1.5)
            ax_res.set_ylabel("Residual", fontsize=8)
            ax_res.set_xlabel("Time (ns)")
            max_res = np.max(np.abs(residual))
            if max_res > 0:
                ax_res.set_ylim(-max_res * 1.1, max_res * 1.1)
            ax_res.grid(True, alpha=0.3)
            ax_res.spines['top'].set_visible(False)
            ax_res.spines['right'].set_visible(False)

        # log both figures in a single step
        self.wandb.log({
            "PMT Visibility": wandb.Image(fig_vis),
            "PDF": wandb.Image(fig_pdf),
            "CDF": wandb.Image(fig_cdf),
        }, step=iteration, commit=False)

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
            self.wandb.log(self._dict, commit=False)
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
        if iteration == 0 or iteration % self._log_every_nsteps != 0:
            return

        if self.is_distributed:
            loss_avg = loss_tensor.detach().clone()
            dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
        else:
            loss_avg = loss_tensor.detach()
        if self.wandb is not None:
            self.wandb.log({
                "loss/avg_across_gpus": loss_avg.item(),
                "loss/rank_0": loss_tensor.item(),
            }, step=iteration, commit=False)

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
        log_dict = {}
        for metric_name, metric_value in metrics_dict.items():
            if not isinstance(metric_value, torch.Tensor):
                metric_value = torch.tensor(metric_value).cuda()

            gathered = [torch.zeros_like(metric_value) for _ in range(self.world_size)]
            dist.all_gather(gathered, metric_value)

            if self.rank == 0:
                for rank_id, val in enumerate(gathered):
                    log_dict[f"{metric_name}/rank_{rank_id}"] = val.item()
                avg_val = torch.stack(gathered).mean()
                log_dict[f"{metric_name}/avg"] = avg_val.item()

        if iteration % self._log_every_nsteps == 0  and self.wandb is not None:
            self.wandb.log(log_dict, step=iteration, commit=False)

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
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 10)
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

