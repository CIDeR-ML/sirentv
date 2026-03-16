from __future__ import annotations
import os
from contextlib import nullcontext
import time

import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from slar.optimizers import optimizer_factory

from sirentv.models import SirenTV
from sirentv.utils.comm import create_ddp_model
from sirentv.utils.log import WandbLogger

from sirentv.data.builder import create_dataloader
from sirentv.infer import build_infer_fn
from sirentv.training.utils import (
    build_losses,
    build_regularizer,
    build_logger,
    compute_loss,
    build_weight_fn,
    backward_step,
    unwrap_net,
)


def train(cfg: dict):
    """
    Unified training loop for all SirenTV modes (waveform, PCA, etc.).

    The loop is 100% generic — mode-specific behavior is resolved at setup time
    via config-driven registries for datasets, models, losses, and inference.
    """
    # --- Distributed setup ---
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{local_rank}")
    else:
        DEVICE = torch.device("cpu")

    iteration_ctr = 0
    epoch_ctr = 0

    # --- Model ---
    net = SirenTV(cfg).to(DEVICE)
    if is_distributed:
        net = create_ddp_model(net, device_ids=[local_rank])
    net = torch.compile(net)

    # --- Data (registry-based, config-driven) ---
    dl = create_dataloader(cfg, rank=rank, world_size=world_size)

    # --- Optimizer ---
    opt, sch, epoch = optimizer_factory(
        list(p for p in net.parameters() if p.requires_grad), cfg
    )
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        if rank == 0:
            print(f"[train] resuming from iteration {iteration_ctr}, epoch {epoch_ctr}")

    # --- Losses, regularizer, logger (all config-driven) ---
    loss_fns = build_losses(cfg)
    regularizer = build_regularizer(cfg)
    logger = build_logger(cfg, net, rank=rank)
    weight_fn = build_weight_fn(cfg)

    # --- Inference/plotting (registry-based, config-driven) ---
    infer = build_infer_fn(cfg, net, dl, DEVICE)

    # --- Save config ---
    if rank == 0:
        with open(os.path.join(logger.logdir, "train_cfg.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    # --- Training hyperparameters ---
    train_cfg = cfg.get("train", {})
    epoch_max = train_cfg.get("max_epochs", int(1e20))
    iteration_max = train_cfg.get("max_iterations", int(1e20))
    save_every_iterations = train_cfg.get("save_every_iterations", -1)
    save_every_epochs = train_cfg.get("save_every_epochs", -1)
    reduction = train_cfg.get("reduction", "mean")

    amp = train_cfg.get("amp", False)
    grad_clip_max_norm = train_cfg.get("clip_grad", None)
    if amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    # Gradient norm logging config
    grad_norm_cfg = cfg.get("logger", {}).get("grad_norm", {})
    log_grad_norm = grad_norm_cfg.get("enabled", False)
    grad_norm_type = grad_norm_cfg.get("norm_type", 2.0)
    grad_norm_per_layer = grad_norm_cfg.get("log_per_layer", False)
    grad_norm_frequency = grad_norm_cfg.get("log_frequency", 1)

    if rank == 0:
        print(f"[train] max iterations {iteration_max}, max epochs {epoch_max}")
        print(f"[train] distributed: {is_distributed}, world_size: {world_size}")

    # --- Training loop ---
    stop_training = False
    losses_display = [float("inf")] * len(loss_fns)

    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        if is_distributed and hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
            dl.sampler.set_epoch(epoch_ctr)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        data_loading_start = time.time()

        pbar = tqdm(dl, desc=f"Epoch {epoch_ctr:<3d}", disable=(rank != 0))
        for batch_idx, data in enumerate(pbar):
            iteration_ctr += 1

            # --- Unpack standardized batch ---
            x = data["position"].to(DEVICE, non_blocking=True)
            target = {k: v.to(DEVICE, non_blocking=True) for k, v in data["target"].items()}
            meta = {k: v.to(DEVICE, non_blocking=True) for k, v in data.get("meta", {}).items()}
            weights = weight_fn(target)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            data_loading_time = time.time() - data_loading_start

            # --- Forward pass ---
            opt.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.time()

            with (torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if amp else nullcontext()):
                fwd_kwargs = {}
                if data.get("return_gradients", False):
                    fwd_kwargs["return_gradients"] = True
                pred = net(x, **fwd_kwargs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_time = time.time() - forward_start

                # --- t0 from PMT positions (only if dataset didn't provide t0) ---
                net_module = unwrap_net(net)
                if hasattr(net_module, "load_pos") and net_module.load_pos and "t0" in pred and "t0" not in target:
                    tick_size = cfg.get("photonlib", {}).get("time_tick_size", 0.1)
                    lAr_r_index = cfg.get("physics", {}).get("R_index", 1.233)
                    speed_of_light = 299.792458 / lAr_r_index
                    pmt_pos = net_module.pmt_coords.to(x.device)
                    pred["t0"] = pred["t0"] * tick_size
                    distances = torch.cdist(x, pmt_pos)
                    target["t0"] = distances / speed_of_light

                # --- Loss computation ---
                losses_dict = compute_loss(pred, target, loss_fns, weights)
                keys, losses_tensor = zip(*losses_dict.items())
                losses_tensor = torch.stack(losses_tensor)

                if reduction == "mean":
                    loss = torch.mean(losses_tensor)
                elif reduction == "geometric_mean":
                    loss = torch.exp(torch.mean(torch.log(losses_tensor)))
                elif reduction == "sum":
                    loss = torch.sum(losses_tensor)
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")

                if regularizer is not None:
                    loss += regularizer(net)

            epoch_loss_sum += loss.detach().item()
            epoch_batch_count += 1

            # --- Backward pass ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_start = time.time()
            backward_step(loss, opt, amp, scaler, grad_clip_max_norm, net)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            model_time_iter = forward_time + backward_time

            # Update display losses
            losses_display = losses_tensor.detach().cpu().tolist()
            pbar.set_postfix_str(", ".join(f"{k}={v:.2e}" for k, v in zip(keys, losses_display)))

            if rank == 0 and iteration_ctr % 10 == 0:
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                print(
                    f"[iter {iteration_ctr:>6d}] "
                    f"loss: {', '.join(f'{k}={v:.2e}' for k, v in zip(keys, losses_display))} | "
                    f"data: {data_loading_time:.3f}s | fwd+bwd: {model_time_iter:.3f}s | "
                    f"mem: {peak_mem:.2f}GB"
                )

            if "grad_mags_transformed" in data or "grad_mag_transformed" in data:
                torch.cuda.empty_cache()

            # --- Logging ---
            if rank == 0:
                current_lr = opt.param_groups[0]["lr"]
                logger.record(
                    ["iter", "epoch", "lr", "data_loading_time", "model_iter_time"]
                    + [f"loss_{k}" for k in keys]
                    + ["loss"],
                    [iteration_ctr, epoch_ctr, current_lr, data_loading_time, model_time_iter]
                    + losses_display
                    + [loss.item()],
                )

            if isinstance(logger, WandbLogger):
                if rank == 0:
                    gpu_mem = float(torch.cuda.memory_allocated(local_rank) / (1024**3)) if torch.cuda.is_available() else 0
                    logger.record(["gpu_memory_gb"], [gpu_mem])

                    if log_grad_norm and iteration_ctr % grad_norm_frequency == 0:
                        logger.log_grad_norms(iteration_ctr, unwrap_net(net), grad_norm_type, grad_norm_per_layer)

                if rank == 0 and iteration_ctr % 10 == 0:
                    with torch.no_grad():
                        inferred = infer(net, x, target, meta)
                    logger.plot(iteration_ctr, inferred)

                    with torch.no_grad():
                        target_log, pred_log = infer.log_step(pred, target, net, meta)
                        logger.step(iteration_ctr, target_log, pred_log)

                if rank == 0:
                    logger.write()
                logger.commit(iteration_ctr)

            # --- Checkpointing ---
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
            epoch_avg_loss = epoch_loss_sum / max(epoch_batch_count, 1)
            sch.step(epoch_avg_loss)

        epoch_ctr += 1

        # Temperature annealing (model-attribute-driven)
        net_module = unwrap_net(net)
        if hasattr(net_module, "anneal_enabled") and net_module.anneal_enabled:
            net_module.anneal_temperature(epoch_ctr, epoch_max)

        if rank == 0 and (save_every_epochs * epoch_ctr) > 0 and epoch_ctr % save_every_epochs == 0:
            filename = os.path.join(
                logger.logdir,
                "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr),
            )
            model_to_save = net.module if is_distributed else net
            model_to_save.save_state(
                filename, opt, sch, iteration_ctr / len(dl), scaler if amp else None
            )

    if rank == 0:
        print(f"[train] Stopped at iteration {iteration_ctr}, epoch {epoch_ctr}")
        infer.cleanup(net, dl, logger, rank)
        logger.write()
        logger.close()

    if is_distributed:
        dist.barrier()


def main():
    import argparse

    is_distributed_env = all(k in os.environ for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])

    if is_distributed_env:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"[main] Initialized distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
    else:
        print("[main] Running in non-distributed mode (single GPU)")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (default: off, uses CSV)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if not args.wandb:
        cfg.setdefault("logger", {})["type"] = "csv"

    train(cfg)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
