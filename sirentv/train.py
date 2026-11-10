from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import yaml
from slar.optimizers import optimizer_factory
from tqdm import tqdm

from sirentv.data.builder import create_dataloader
from sirentv.infer import build_infer_fn
from sirentv.models import SirenTV
from sirentv.training.utils import (
    backward_step,
    build_logger,
    build_losses,
    build_weight_fn,
    compute_loss,
    resolve_config,
    unwrap_net,
)
from sirentv.utils.comm import create_ddp_model
from sirentv.utils.log import WandbLogger


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _grad_supervision_warmup(train_cfg):
    """train.grad_supervision_keys as {key: start}: a list means start at 0, a dict gives per-key starts.

    A start delay matters for coeffs/quantiles: the aggregate estimator weights by the model's
    own predicted visibility, which is meaningless until v has trained.
    """
    keys_cfg = train_cfg.get("grad_supervision_keys", [])
    if isinstance(keys_cfg, dict):
        return {k: int(v) for k, v in keys_cfg.items()}
    return {k: 0 for k in keys_cfg}


def train(cfg: dict):
    """Training loop shared by every mode; datasets, models, losses and inference come from the config."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    DEVICE = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # "high" = TF32 matmuls (fast on A100); "highest" = strict fp32. Runs are not bit-comparable across settings.
    matmul_precision = cfg.get("device", {}).get("matmul_precision", "high")
    torch.set_float32_matmul_precision(matmul_precision)
    if rank == 0:
        print(f"[train] float32 matmul precision: {matmul_precision}")

    iteration_ctr = 0
    epoch_ctr = 0

    # train.seed unset = unseeded weight init
    seed = cfg.get("train", {}).get("seed")
    if seed is not None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if rank == 0:
            print(f"[train] seed set to {seed}")

    net = SirenTV(cfg).to(DEVICE)
    if is_distributed:
        net = create_ddp_model(net, device_ids=[local_rank])

    train_cfg = cfg.get("train", {})
    if train_cfg.get("grad_supervision_keys", []):
        if rank == 0:
            print("[train] grad_supervision_keys configured -- skipping torch.compile "
                  "(AOTAutograd does not support the double backward it needs)")
    else:
        net = torch.compile(net)

    dl = create_dataloader(cfg, rank=rank, world_size=world_size)

    opt, sch, epoch = optimizer_factory(
        list(p for p in net.parameters() if p.requires_grad), cfg
    )
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        if rank == 0:
            print(f"[train] resuming from iteration {iteration_ctr}, epoch {epoch_ctr}")
        # restore the lr the scheduler implies at this epoch
        if sch is not None:
            restored = sch.get_last_lr()
            for group, lr in zip(opt.param_groups, restored):
                group["lr"] = lr
            if rank == 0:
                print(f"[train] restored scheduler lr: {restored}")

    loss_fns = build_losses(cfg)
    logger = build_logger(cfg, net, rank=rank)
    weight_fn = build_weight_fn(cfg)
    infer = build_infer_fn(cfg, net, dl, DEVICE)

    if rank == 0:
        with open(os.path.join(logger.logdir, "train_cfg.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    # max_epochs counts epochs in THIS session, on top of a resumed epoch
    epoch_max = epoch_ctr + train_cfg.get("max_epochs", int(1e20))
    iteration_max = train_cfg.get("max_iterations", int(1e20))
    save_every_iterations = train_cfg.get("save_every_iterations", -1)
    save_every_epochs = train_cfg.get("save_every_epochs", -1)
    reduction = train_cfg.get("reduction", "mean")

    amp = train_cfg.get("amp", False)
    grad_clip_max_norm = train_cfg.get("clip_grad", None)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    grad_norm_cfg = cfg.get("logger", {}).get("grad_norm", {})
    log_grad_norm = grad_norm_cfg.get("enabled", False)
    grad_norm_type = grad_norm_cfg.get("norm_type", 2.0)
    grad_norm_per_layer = grad_norm_cfg.get("log_per_layer", False)
    grad_norm_frequency = grad_norm_cfg.get("log_frequency", 1)

    plot_every_nsteps = cfg.get("logger", {}).get("plot_every_nsteps")
    if plot_every_nsteps is None or int(plot_every_nsteps) <= 0:
        plot_every_nsteps = max(1, len(dl))
    plot_every_nsteps = int(plot_every_nsteps)
    if rank == 0:
        print(f"[train] diagnostics/plot cadence: every {plot_every_nsteps} iterations "
              f"({len(dl)} iterations/epoch)")

    net_module = unwrap_net(net)

    # Keys whose `{key}_grad_frob` the model also computes, for WeightedGradFrobLoss.
    grad_supervision_warmup = _grad_supervision_warmup(train_cfg)
    grad_supervision_warmup_unit = train_cfg.get("grad_supervision_warmup_unit", "epoch")
    assert grad_supervision_warmup_unit in ("epoch", "iteration"), (
        f"grad_supervision_warmup_unit must be 'epoch' or 'iteration', got {grad_supervision_warmup_unit!r}"
    )
    # per-PMT (default) costs O(n_pmt) backward passes and can OOM; aggregate is O(1)
    grad_supervision_aggregate = bool(train_cfg.get("grad_supervision_aggregate", False))
    _grad_supervision_announced = set()

    if rank == 0:
        print(f"[train] max iterations {iteration_max}, max epochs {epoch_max}")
        print(f"[train] distributed: {is_distributed}, world_size: {world_size}")

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

            x = data["position"].to(DEVICE, non_blocking=True)
            target = {k: v.to(DEVICE, non_blocking=True) for k, v in data["target"].items()}
            meta = {k: v.to(DEVICE, non_blocking=True) for k, v in data.get("meta", {}).items()}
            weights = weight_fn(target)

            _sync()
            data_loading_time = time.time() - data_loading_start

            opt.zero_grad()
            _sync()
            forward_start = time.time()

            with (torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if amp else nullcontext()):
                fwd_kwargs = {}
                _warmup_progress = epoch_ctr if grad_supervision_warmup_unit == "epoch" else iteration_ctr
                active_grad_keys = [k for k, warmup in grad_supervision_warmup.items() if _warmup_progress >= warmup]
                if rank == 0:
                    for k in active_grad_keys:
                        if k not in _grad_supervision_announced:
                            _grad_supervision_announced.add(k)
                            print(
                                f"[train] gradient supervision on '{k}' starts now "
                                f"(warmup={grad_supervision_warmup[k]} {grad_supervision_warmup_unit}s passed)"
                            )
                if active_grad_keys:
                    fwd_kwargs["grad_keys"] = active_grad_keys
                    fwd_kwargs["grad_aggregate"] = grad_supervision_aggregate
                pred = net(x, **fwd_kwargs)

                _sync()
                forward_time = time.time() - forward_start

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

            epoch_loss_sum += loss.detach().item()
            epoch_batch_count += 1

            _sync()
            backward_start = time.time()
            backward_step(loss, opt, amp, scaler, grad_clip_max_norm, net)
            _sync()
            model_time_iter = forward_time + time.time() - backward_start

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
                        logger.log_grad_norms(iteration_ctr, net_module, grad_norm_type, grad_norm_per_layer)

                if rank == 0 and iteration_ctr % plot_every_nsteps == 0:
                    with torch.no_grad():
                        inferred = infer(net, x, target, meta)
                    logger.plot(iteration_ctr, inferred)

                    with torch.no_grad():
                        target_log, pred_log = infer.log_step(pred, target, net, meta)
                        logger.step(iteration_ctr, target_log, pred_log)

                if rank == 0:
                    logger.write()
                logger.commit(iteration_ctr)
            elif rank == 0 and iteration_ctr % logger.log_every_nsteps == 0:
                # CSV: one row (losses, lr, analysis metrics) every logger.log_every_nsteps
                with torch.no_grad():
                    target_log, pred_log = infer.log_step(pred, target, net, meta)
                logger.step(iteration_ctr, target_log, pred_log)

            if rank == 0 and save_every_iterations > 0 and iteration_ctr % save_every_iterations == 0:
                filename = os.path.join(
                    logger.logdir,
                    "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr),
                )
                net_module.save_state(filename, opt, sch, iteration_ctr, scaler if amp else None)

            if iteration_max <= iteration_ctr:
                stop_training = True
                break

            _sync()
            data_loading_start = time.time()

        if stop_training:
            break

        if sch is not None:
            epoch_avg_loss = epoch_loss_sum / max(epoch_batch_count, 1)
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(epoch_avg_loss)
            else:
                sch.step()

        epoch_ctr += 1

        if rank == 0 and (save_every_epochs * epoch_ctr) > 0 and epoch_ctr % save_every_epochs == 0:
            filename = os.path.join(
                logger.logdir,
                "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr),
            )
            net_module.save_state(
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
    parser.add_argument("--config", type=str, required=True,
                        help="Config path, or a name in config/ (config/archive/ with --archive)")
    parser.add_argument("--archive", action="store_true", default=False,
                        help="Read --config by name from config/archive/")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Log to Weights & Biases (default: CSV)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Set train.resume=True (restore optimizer/scheduler/epoch from model.ckpt_file)")
    args = parser.parse_args()

    with open(resolve_config(args.config, args.archive)) as f:
        cfg = yaml.safe_load(f)

    if not args.wandb:
        cfg.setdefault("logger", {})["type"] = "csv"

    if args.resume:
        cfg.setdefault("train", {})["resume"] = True

    train(cfg)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
