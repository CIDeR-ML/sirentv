import os
from functools import partial
import glob
import numpy as np
import torch

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

    def __init__(self, cfg, rank=0):
        """
        Constructor

        Parameters
        ----------
        cfg : dict
            A collection of configuration parameters. 'project' and 'name' specify
            the wandb project and run name respectively.
        rank : int
            Global rank of this process
        """
        self.rank = rank

        log_cfg = cfg.get("logger", dict())
        self.project = log_cfg.get("project", "default-project")
        self.name = log_cfg.get("name", None)
        self.entity = log_cfg.get("entity", None)
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 1)

        if self.rank == 0:
            run_dir = log_cfg.get("run_dir", "./")
            dir_name = log_cfg.get("dir_name", "logs")
            if not os.path.isabs(dir_name):
                dir_name = os.path.join(run_dir, dir_name)
            self._logdir = self.make_logdir(dir_name)
            self._wandb_rundir = run_dir
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
                config=cfg,
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

        if self.rank != 0:
            return

        if None not in (label, pred) and len(self._analysis_dict) > 0:
            local_metrics = {}
            for key, f in self._analysis_dict.items():
                try:
                    metric_value = f(label, pred)
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.detach().item() if metric_value.numel() == 1 else metric_value.detach()
                    local_metrics[f"analysis/{key}"] = metric_value
                except Exception as e:
                    print(f"[WandbLogger] Warning: Failed to compute {key}: {e}")

            if self.wandb is not None:
                self.wandb.log(local_metrics, step=iteration, commit=False)

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

        if self.rank != 0 or inferred is None:
            return

        if iteration % self._log_every_nsteps != 0 or self.wandb is None:
            return

        data = {
            'x_value': inferred['x_value'].detach().cpu(),
            'visibility': inferred['visibility'].detach().cpu(),
            'pdf': inferred['pdf'].detach().cpu(),
            'cdf': inferred['cdf'].detach().cpu(),
        }
        if 'position' in inferred and inferred['position'] is not None:
            data['position'] = inferred['position'].detach().cpu()
        else:
            data['position'] = None
        if 't0' in inferred and inferred['t0'] is not None:
            data['t0'] = inferred['t0'].detach().cpu()
        else:
            data['t0'] = None

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Build subtitle from position
        subtitle = ""
        if data['position'] is not None:
            pos = data['position'].numpy()
            if pos.ndim > 1:
                pos = pos[0]
            subtitle = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"

        # --- Visibility plot ---
        fig_vis, ax = plt.subplots(figsize=(6, 5))

        visibility = data['visibility'].numpy()
        v_target = visibility[:, 0]
        v_pred = visibility[:, 1]
        pos_mask = (v_target > 0) & (v_pred > 0)
        if not pos_mask.any():
            # v_target is fixed ground truth for this diagnostic voxel, so this can only mean
            # v_pred is non-positive (or NaN, which also fails ">0") for EVERY PMT here -- a
            # real, worth-knowing-about model issue, but this plot is a monitoring aid, not the
            # training loop itself: crashing the whole run (and losing everything trained so
            # far) over a diagnostic plot failing is strictly worse than just skipping this one
            # plot and letting training continue.
            print(f"[WandbLogger.plot] iteration {iteration}: skipping visibility plot, "
                  f"no (target, pred) pair is both positive (v_pred range "
                  f"[{np.nanmin(v_pred):.3g}, {np.nanmax(v_pred):.3g}] -- check for NaN/collapse)")
            ax.set_title(f"PMT Visibility (skipped: v_pred non-positive/NaN for every PMT)")
        else:
            ax.scatter(v_target[pos_mask], v_pred[pos_mask], s=10, alpha=0.6, zorder=3)

            vmin = min(v_target[pos_mask].min(), v_pred[pos_mask].min()) * 0.5
            vmax = max(v_target[pos_mask].max(), v_pred[pos_mask].max()) * 2.0
            t_line = np.array([vmin, vmax])
            ax.plot(t_line, t_line, 'k-', alpha=0.8, lw=1, label="y=x")
            for bias_pct, color, ls in [(5, "green", "--"), (10, "orange", ":")]:
                b = bias_pct / 100.0
                upper = t_line * (2 + b) / (2 - b)
                lower = t_line * (2 - b) / (2 + b)
                ax.fill_between(t_line, lower, upper, alpha=0.10, color=color,
                                label=f"\u00b1{bias_pct}% bias")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel("Target Visibility")
            ax.set_ylabel("Predicted Visibility")
            ax.set_title(f"PMT Visibility\n{subtitle}" if subtitle else "PMT Visibility")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # --- PDF plot ---
        fig_pdf, ax = plt.subplots(figsize=(6, 5))

        x = data['x_value'].numpy()
        pdf = data['pdf'].numpy()

        mask = x > 0
        x_pos = x[mask]
        pdf_pos = pdf[mask]
        floor = 1e-30
        pdf_target = np.clip(pdf_pos[:, 0], floor, None)
        pdf_pred = np.clip(pdf_pos[:, 1], floor, None)

        ax.plot(x_pos, pdf_target, label="Target", color="navy", linewidth=2)
        ax.plot(x_pos, pdf_pred, label="Predicted", color="darkorange", linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-5)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("PDF")
        ax.set_title(f"PDF\n{subtitle}" if subtitle else "PDF")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        # --- CDF plot with residual ---
        fig_cdf = plt.figure(figsize=(6, 6))
        gs = GridSpec(5, 1, figure=fig_cdf, hspace=0.4)
        ax_cdf = fig_cdf.add_subplot(gs[0:4, 0])
        ax_res = fig_cdf.add_subplot(gs[4, 0], sharex=ax_cdf)

        cdf = data['cdf'].numpy()
        ax_cdf.plot(x, cdf[:, 0], label="Target", color="navy", linewidth=2)
        ax_cdf.plot(x, cdf[:, 1], label="Predicted", color="darkorange", linewidth=2)

        if data['t0'] is not None:
            t0s = data['t0'].numpy()
            ax_cdf.axvline(t0s[0], linestyle='--', label="Target T0", color="navy", alpha=0.6)
            ax_cdf.axvline(t0s[1], linestyle='--', label="Predicted T0", color="darkorange", alpha=0.6)

        ax_cdf.set_ylabel("Value")
        ax_cdf.set_title(f"CDF - {subtitle}" if subtitle else "CDF", fontsize=10)
        ax_cdf.legend(fontsize=8)
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.tick_params(labelbottom=False)

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
    
    def log_grad_norms(self, iteration, net, norm_type=2.0, log_per_layer=False):
        """Log gradient norms to wandb. Call after backward() on rank 0 only.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        net : nn.Module
            The model (unwrapped, not DDP wrapper).
        norm_type : float
            Norm type (default 2.0 for L2).
        log_per_layer : bool
            If True, also log per-layer gradient norms.
        """
        if self.wandb is None:
            return

        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if len(grads) == 0:
            return

        if norm_type == float('inf'):
            total_norm = max(g.abs().max() for g in grads)
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(g, norm_type) for g in grads]),
                norm_type,
            )

        log_dict = {"grad_norm/total": total_norm.item()}

        if log_per_layer:
            for name, param in net.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad, norm_type)
                    clean_name = name.replace('.', '/')
                    log_dict[f"grad_norm/layers/{clean_name}"] = grad_norm.item()

        self.wandb.log(log_dict, step=iteration, commit=False)

class CSVLogger(Logger):
    """
    Logger class to store training progress in a CSV file.
    """

    def __init__(self, cfg, rank=0):
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
        """

        self.rank = rank

        log_cfg = cfg.get("logger", dict())
        self._log_every_nsteps = log_cfg.get("log_every_nsteps", 10)
        if self.rank == 0:
            run_dir = log_cfg.get("run_dir", "./")
            dir_name = log_cfg.get("dir_name", "logs")
            if not os.path.isabs(dir_name):
                dir_name = os.path.join(run_dir, dir_name)
            self._logdir = self.make_logdir(dir_name)
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
            analysis_cfg = log_cfg.get("analysis", [])
            if isinstance(analysis_cfg, dict):
                analysis_cfg = [{"func": k, **v} for k, v in analysis_cfg.items()]
            for item in analysis_cfg:
                item = dict(item)
                key = item.pop("func")
                print("[CSVLogger] adding analysis function:", key)
                suffix = item.pop("suffix", "")
                if suffix:
                    suffix = f"_{suffix}"
                self._analysis_dict[key + suffix] = partial(
                    getattr(importlib.import_module("sirentv.analysis"), key), **item
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

