from __future__ import annotations
import os
from contextlib import nullcontext
from typing import Literal
import time


import torch
import torch.nn as nn
import torch.distributed as dist

import yaml
from sirentv.data.io import PLibDataLoader
from sirentv.data.io import create_dataloader

from slar.optimizers import optimizer_factory
from slar.utils import get_device
from tqdm import tqdm

from sirentv.analysis import get_pred_target, log_imshow, log_line, log_pred_target

from sirentv.models import SirenTV
from sirentv.loss.builder import build_loss as build_loss_fn, build_regularizer as build_regularizer_fn
from sirentv.weighting import build_weighting
from sirentv.utils.comm import create_ddp_model
from sirentv.utils.log import CSVLogger, WandbLogger, Logger
from sirentv.utils.transform import pdf_to_cdf
from sirentv.infer import infer_single_pos_single_pmt

def build_weightings(cfg):
    """Build weighting modules from config."""
    weight_cfg = cfg.get("data", {}).get("weight", {})
    weightings = {}
    for key, wcfg in weight_cfg.items():
        if wcfg is None or not wcfg.get("enable", True):
            weightings[key] = None
        else:
            weightings[key] = build_weighting(wcfg)
    return weightings

def build_losses(cfg):
    loss_cfg = cfg.get("train", dict()).get("loss", [])
    losses = []
    for cfg in loss_cfg:
        losses.append(build_loss_fn(cfg))
    return losses

def build_regularizer(cfg) -> nn.Module | None:
    regularizer_cfg = cfg.get("train", dict()).get("regularization", None)
    if regularizer_cfg is None:
        return None
    return build_regularizer_fn(regularizer_cfg)

def build_logger(cfg, net, rank=0, world_size=1, is_distributed=False) -> Logger:
    logger_type = cfg.get("logger", dict()).get("type", "csv")
    if logger_type == "csv":
        logger = CSVLogger(cfg, rank=rank, world_size=world_size, is_distributed=is_distributed)
    else:
        logger = WandbLogger(cfg, rank=rank, world_size=world_size, is_distributed=is_distributed)
    if hasattr(logger, "watch_grad"):
        logger.watch_grad(net)
    return logger


def compute_loss(
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        losses: list[nn.Module],
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
    losses_out = {}
    for loss in losses:
        curr_loss = loss(pred, target, weights)
        cls_name = loss.__class__.__name__.lower()
        losses_out[f"{cls_name}_{loss.key}"] = curr_loss
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
    # Initialize distributed process group
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # Set device to local rank
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{local_rank}")
    else:
        DEVICE = torch.device("cpu")

    #print(f"[DEBUG] Rank {rank}/{world_size}, is_distributed={is_distributed}, "
    #      f"dist.is_initialized()={dist.is_initialized()}")

    iteration_ctr = 0
    epoch_ctr = 0

    # Create necessary pieces: the model, optimizer, loss, logger.
    # Load the states if this is resuming.
    net = SirenTV(cfg).to(DEVICE)
    if is_distributed:
        net = create_ddp_model(
            net,
            device_ids=[local_rank],
        )
    mode: Literal["pdf", "cdf"] = (net.module if is_distributed else net).mode

    #dl = PLibDataLoader(cfg, device=DEVICE, rank=rank, world_size=world_size)
    dl = create_dataloader(cfg, rank=rank, world_size=world_size)

    opt, sch, epoch = optimizer_factory(list(p for p in net.parameters() if p.requires_grad), cfg)
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        if rank == 0:
            print(
                "[train] resuming training from iteration",
                iteration_ctr,
                "epoch",
                epoch_ctr,
            )

    loss_fns = build_losses(cfg)
    regularizer = build_regularizer(cfg)
    weightings = build_weightings(cfg)
    logger = build_logger(cfg, net, rank=rank, world_size=world_size, is_distributed=is_distributed)

    # Store configuration (only on rank 0)
    if rank == 0:
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
    if rank == 0:
        print(f"[train] train for max iterations {iteration_max} or max epochs {epoch_max}")
        print(f"[train] distributed training: {is_distributed}, world_size: {world_size}")

    # Start the training loop
    stop_training = False
    losses = [float('inf')] * len(loss_fns)

    lAr_r_index = cfg.get("physics", {}).get("R_index", 1.233)
    speed_of_light = 299.792458/lAr_r_index # mm/ns

    tick_size = cfg.get("photonlib", {}).get("time_tick_size", 0.1)

    # through epochs
    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        if is_distributed and hasattr(dl, 'set_epoch'):
            dl.set_epoch(epoch_ctr)

        # through batches
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        data_loading_start = time.time()
        for batch_idx, data in enumerate(tqdm(dl, desc="Epoch %-3d; Loss %-3s" % (epoch_ctr, ",".join(["%.2e" % l for l in losses])), disable=(rank != 0))):
            iteration_ctr += 1

            with (torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if amp else nullcontext()):
                x = data["position"].contiguous().to(DEVICE, non_blocking=True)
                target_t_pdf = data["target"].contiguous().to(DEVICE, non_blocking=True)
                target_t_pdf_linear = data["target_linear"].contiguous().to(DEVICE, non_blocking=True)
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

                # generate weights using weighting modules
                weights = {
                    k: weightings[k](target[k]) if k in weightings and weightings[k] is not None else 1.0
                    for k in target.keys()
                }
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                data_loading_time = time.time() - data_loading_start

                # Running the model, compute the loss, back-prop gradients to optimize.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_start = time.time()
                pred: dict[str, torch.Tensor] = net.module(x) if is_distributed else net(x)
                # OUTPUTS:
                # v: visibilities, (B, N_pmt)
                # t: CDF/PDF, (B, N_pmt, N_time)
                # t0 (possibly, in the units of ticks)
                mem_after_forward = torch.cuda.memory_allocated()/(1024**3) if torch.cuda.is_available() else 0 # in GB

                if hasattr(net.module if is_distributed else net, 'load_pos'):
                    load_pos = (net.module if is_distributed else net).load_pos
                    if load_pos:
                        net_module = net.module if is_distributed else net
                        pmt_pos = net_module.pmt_coords.to(x.device)
                        pred['t0'] *= tick_size
                        distances = torch.cdist(x, pmt_pos)
                        tof = distances / speed_of_light # in ns
                        target["t0"] = tof.cpu()

                losses = compute_loss(
                    pred,
                    target,
                    loss_fns,
                    weights,
                )

                keys, losses = zip(*losses.items())
                losses = torch.stack(losses)

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
                torch.cuda.synchronize()
                model_time_iter = time.time() - forward_start

            if rank == 0:
                # get current learning rate
                current_lr = opt.param_groups[0]['lr']
                # Log training parameters
                logger.record(
                    ["iter", "epoch", "lr", "data_loading_time", "model_iter_time", "model_forward_mem_usage"] + [f'loss_{k}' for k in keys] + ["loss"],
                    [iteration_ctr, epoch_ctr, current_lr, data_loading_time, model_time_iter, mem_after_forward] + losses.detach().cpu().tolist() + [loss.item()],
                )

                # Step the logger
                with torch.no_grad():
                    pred['v_linear'] = dl.inv_xform_vis(pred['v'])
                    pred['t_linear'] = pred['t']
                    logger.step(iteration_ctr, target, pred)

            if isinstance(logger, WandbLogger):
                logger.log_aggregated_loss(iteration_ctr, loss)

            if iteration_ctr % 10 == 0 and isinstance(logger, WandbLogger):
                per_rank_metrics = {
                    "gpu_memory_gb": float(torch.cuda.memory_allocated(local_rank)/(1024**3)) if torch.cuda.is_available() else 0,
                    "loss": loss.detach(),
                    "data_loading_time": data_loading_time,
                    "model_forward_time": model_time_iter,
                }
                logger.log_per_rank_metrics(iteration_ctr, per_rank_metrics)

            if rank == 0 and iteration_ctr % 10 == 0:
                inferred_output = infer_single_pos_single_pmt(net.module if is_distributed else net, x, target, tick_size)
                logger.plot(iteration_ctr, inferred_output)

            # Save the model parameters if the condition is met
            if rank == 0 and save_every_iterations > 0 and iteration_ctr % save_every_iterations == 0:
                filename = os.path.join(
                    logger.logdir,
                    "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr),
                )
                model_to_save = net.module if is_distributed else net
                model_to_save.save_state(filename, opt, sch, iteration_ctr, scaler if amp else None)

            if iteration_max <= iteration_ctr:
                stop_training = True
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            data_loading_start = time.time()

        if stop_training:
            break

        if sch is not None:
            sch.step(loss)

        epoch_ctr += 1

        if rank == 0 and (save_every_epochs * epoch_ctr) > 0 and epoch_ctr % save_every_epochs == 0:
            filename = os.path.join(
                logger.logdir, "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr)
            )
            model_to_save = net.module if is_distributed else net
            model_to_save.save_state(
                filename,
                opt,
                sch,
                iteration_ctr / len(dl),
                scaler if amp else None,
            )
            # logger.save(filename)

    if rank == 0:
        print("[train] Stopped training at iteration", iteration_ctr, "epochs", epoch_ctr)

        # logging after training.
        logger.write()
        pred, target = get_pred_target(dl, net.module if is_distributed else net)
        for k in pred.keys():
            log_pred_target(pred[k], target[k], name=f"comparison_{k}")
        logger.close()

    if is_distributed:
        dist.barrier()


def main():
    import argparse
    # Debug environment before anything else
    #print(f"[DEBUG] Environment variables:")
    #print(f"  RANK: {os.environ.get('RANK', 'NOT SET')}")
    #print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    #print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    #print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    #print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    #print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    #print(f"  Available CUDA devices: {torch.cuda.device_count()}")

    # only initialize DDP if environment variables indicate it should be used
    ddp_enabled = 'RANK' in os.environ or 'WORLD_SIZE' in os.environ
    if ddp_enabled:
        dist.init_process_group(backend='nccl')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    default_config_path = '/sdf/home/y/youngsam/sw/dune/siren-t/config/siren_4848-bivis.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    train(cfg)
    
    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
