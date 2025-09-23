from __future__ import annotations
import os
import time
from contextlib import nullcontext
from typing import Literal, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from sirentv.data.io import PLibDataLoader
from slar.optimizers import optimizer_factory
from slar.utils import get_device
from tqdm import tqdm

from sirentv.analysis import get_pred_target, log_imshow, log_line, log_pred_target

from sirentv.models import SirenTV
from sirentv.loss.builder import build_loss as build_loss_fn, build_regularizer as build_regularizer_fn
from sirentv.utils.comm import create_ddp_model
from sirentv.utils.log import CSVLogger, WandbLogger
from sirentv.utils.transform import pdf_to_cdf

def get_weight_by_vis(vis, factor=None, threshold=1e-8):
    """
    Weight by visibility, `weight  = vis * factor`.
    Weights (after applying factor) below `threshold` are set to 1.

    Arguments
    ---------
    vis: torch.Tensor
        Visibility values.

    Returns
    -------
    w: torch.Tensor
        Weight values with `w.shape == vis.shape`.
    """
    if factor is None:
        factor = 1 / torch.max(vis.clamp(min=1e-8))
    w = vis * factor
    w[w < threshold] = 1.0
    return w

def build_losses(cfg):
    loss_cfg = cfg.get("train", dict()).get("loss", [])
    losses = []
    for cfg in loss_cfg:
        losses.append(build_loss_fn(cfg))
    return losses

def build_regularizer(cfg):
    regularizer_cfg = cfg.get("train", dict()).get("regularization", None)
    if regularizer_cfg is None:
        return None
    return build_regularizer_fn(regularizer_cfg)

def build_logger(cfg, net):
    logger_type = cfg.get("logger", dict()).get("type", "csv")
    logger = CSVLogger(cfg) if logger_type == "csv" else WandbLogger(cfg)
    if hasattr(logger, "watch_grad"):
        logger.watch_grad(net)
    return logger


def compute_loss(
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        losses: list[nn.Module],
        weights: dict[str, torch.Tensor],
    ):
    losses_out = []
    for loss in losses:
        curr_loss = loss(pred, target, weights)
        losses_out.append(curr_loss)
    losses_out = torch.stack(losses_out)
    return losses_out

def train(cfg: dict):
    """
    A function to run an optimization loop for SirenVis model.
    Configuration specific to this function is "train" at the top level.

    Parameters
    ----------
    max_epochs : int
        The maximum number of epochs before stopping training

    max_iterations : int
        The maximum number of iterations before stopping training

    save_every_epochs : int
        A period in epochs to store the network state

    save_every_iterations : int
        A period in iterations to store the network state

    optimizer_class : str
        An optimizer class name to train SirenVis

    optimizer_param : dict
        Optimizer constructor arguments

    resume : bool
        If True, and if a checkopint file is provided for the model, resume training
        with the optimizer state restored from the last checkpoint step.

    """

    # Initialize wandb
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.get("device"):
        DEVICE = get_device(cfg["device"]["type"])

    iteration_ctr = 0
    epoch_ctr = 0

    # Create necessary pieces: the model, optimizer, loss, logger.
    # Load the states if this is resuming.
    net = SirenTV(cfg)
    # net = create_ddp_model(net) # TODO: add ddp
    dl = PLibDataLoader(cfg, device=DEVICE)
    mode: Literal["pdf", "cdf"] = net.mode

    net.to(DEVICE)
    opt, sch, epoch = optimizer_factory(list(p for p in net.parameters() if p.requires_grad), cfg)
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        print(
            "[train] resuming training from iteration",
            iteration_ctr,
            "epoch",
            epoch_ctr,
        )

    loss_fns = build_losses(cfg)
    regularizer = build_regularizer(cfg)
    logger = build_logger(cfg, net)

    # Store configuration   
    with open(os.path.join(logger.logdir, "train_cfg.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # Set the control parameters for the training loop
    train_cfg = cfg.get("train", dict())
    epoch_max = train_cfg.get("max_epochs", int(1e20))
    iteration_max = train_cfg.get("max_iterations", int(1e20))
    save_every_iterations = train_cfg.get("save_every_iterations", -1)
    save_every_epochs = train_cfg.get("save_every_epochs", -1)
    reduction = cfg.get("train", dict()).get("reduction", "mean")

    # amp flag
    amp = train_cfg.get("amp", False)
    if amp:
        scaler = torch.amp.GradScaler('cuda')
    print(f"[train] train for max iterations {iteration_max} or max epochs {epoch_max}")

    weight_cfg = cfg.get("data", {}).get("weight", {})

    # Start the training loop
    stop_training = False
    losses = [float('inf')] * len(loss_fns)

    # through epochs
    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        # through batches
        for batch_idx, data in enumerate(tqdm(dl, desc="Epoch %-3d; Loss %-3s" % (epoch_ctr, ",".join(["%.2e" % l for l in losses])))):
            iteration_ctr += 1
            with (torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if amp else nullcontext()):

                x = data["position"].contiguous()#.to(DEVICE)
                target_t_pdf = data["target"].contiguous().squeeze()#.to(DEVICE)
                target_t_pdf_linear = data["target_linear"].contiguous().squeeze()#.to(DEVICE)
                target_v_linear = target_t_pdf_linear.sum(-1)
                target_v = dl.xform_vis(target_v_linear)

                if mode == "cdf":
                    target_t_cdf = pdf_to_cdf(target_t_pdf_linear) # <-- in linear domain!

                target = {
                    "t": target_t_cdf if mode == "cdf" else target_t_pdf,
                    # output for cdf is in linear domain already!
                    "t_linear": target_t_pdf_linear if mode == "pdf" else target_t_cdf,
                    "v": target_v,
                    "v_linear": target_v_linear,
                }

                # generate weights for just v! (and for t if mode==pdf; enable via config)
                weights = {
                    k: get_weight_by_vis(
                        target[k],
                        factor=weight_cfg[k].get("factor", None),
                        threshold=weight_cfg[k].get("threshold", 1e-8),
                    )
                    if (k in weight_cfg and weight_cfg[k]['enable']) else 1.0
                    for k in target.keys()
                }

                # Running the model, compute the loss, back-prop gradients to optimize.
                pred: dict[str, torch.Tensor] = net(x)
                # OUTPUTS:
                # v: visibilities, (B, N_pmt)
                # t: CDF/PDF, (B, N_pmt, N_time)
                # t0 (possibly)

                losses = compute_loss(
                    pred,
                    target,
                    loss_fns,
                    weights,
                )

                if reduction == "mean":
                    loss = torch.mean(losses)
                elif reduction == "geometric_mean":
                    loss = torch.exp(torch.mean(torch.log(losses)))
                elif reduction == "sum":
                    loss = torch.sum(losses)
                else:
                    raise ValueError(f"Unknown reduction method: {reduction} not in [mean, geometric_mean, sum]")

                if regularizer is not None:
                    loss += regularizer(net)

                opt.zero_grad()
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

            # get current learning rate
            current_lr = opt.param_groups[0]['lr']

            # Log training parameters           
            logger.record(
                ["iter", "epoch", "lr"] + [f'loss_{i}' for i in range(len(losses))] + ["loss"],
                [iteration_ctr, epoch_ctr, current_lr] + losses.detach().cpu().tolist() + [loss.item()],
            )

            # Step the logger
            with torch.no_grad():
                pred['t_linear'] = dl.inv_xform_vis(pred['t'])
                logger.step(iteration_ctr, target, pred)

            # Save the model parameters if the condition is met
            if save_every_iterations > 0 and iteration_ctr % save_every_iterations == 0:
                filename = os.path.join(
                    logger.logdir,
                    "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr),
                )
                net.save_state(filename, opt, sch, iteration_ctr, scaler if amp else None)

            if iteration_max <= iteration_ctr:
                stop_training = True
                break

        if stop_training:
            break

        if sch is not None:
            sch.step(loss)

        epoch_ctr += 1

        if (save_every_epochs * epoch_ctr) > 0 and epoch_ctr % save_every_epochs == 0:
            filename = os.path.join(
                logger.logdir, "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr)
            )
            net.save_state(
                filename,
                opt,
                sch,
                iteration_ctr / len(dl),
                scaler if amp else None,
            )
            # logger.save(filename)


    print("[train] Stopped training at iteration", iteration_ctr, "epochs", epoch_ctr)

    # logging after training.
    logger.write()
    pred, target = get_pred_target(dl, net)
    for k in pred.keys():
        log_pred_target(pred[k], target[k], name=f"comparison_{k}")
    logger.close()


def main():
    import argparse

    import yaml
    
    
    default_config_path = '/sdf/home/y/youngsam/sw/dune/siren-t/config/siren_4848-bivis.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    train(cfg)

    


if __name__ == "__main__":
    main()