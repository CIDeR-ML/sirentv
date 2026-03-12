"""PCA-mode training loop for SirenTV with compressed photon library."""
from __future__ import annotations

import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm
from slar.optimizers import optimizer_factory

from sirentv.data.compressed import CompressedPLib, create_compressed_dataloader
from sirentv.models import SirenTV
from sirentv.train import (
    build_logger,
    build_losses,
    build_regularizer,
    compute_loss,
    get_weight_by_vis,
)
from sirentv.utils.comm import create_ddp_model
from sirentv.utils.log import WandbLogger
from sirentv.utils.transform import cdf_to_pdf


def _infer_pca_plot(net, x, target, cplib, batch_id=0, pmt_id=40, denorm=False):
    """Reconstruct CDF/PDF from PCA predictions for WandB plotting.

    Returns a dict matching the format expected by WandbLogger.plot().
    """
    net_module = net.module if hasattr(net, "module") else net
    # unwrap torch.compile to avoid dynamo re-tracing with a different batch size
    if hasattr(net_module, "_orig_mod"):
        net_module = net_module._orig_mod
    net_module.freeze_all()

    with torch.no_grad():
        pred = net_module(x)
    pred_coeffs = pred["t"]     # (B, N_pmt, n_components)
    pred_log_t0 = pred["t0"]    # (B, N_pmt)
    pred_v = pred["v"]          # (B, N_pmt)

    # Inverse transform visibility for comparison
    pred_v_linear = net_module._inv_xform_vis(pred_v[batch_id, :])
    target_v_linear = net_module._inv_xform_vis(target["v"][batch_id, :].to(pred_v.device))

    # Denormalize coefficients back to raw PCA space for CDF reconstruction
    pred_c = pred_coeffs[batch_id:batch_id+1, :, :]
    target_c = target["coeffs"][batch_id:batch_id+1, :, :].to(pred_coeffs.device)
    if denorm and cplib.coeff_std is not None:
        pred_c = cplib.denormalize_coeffs(pred_c)
        target_c = cplib.denormalize_coeffs(target_c)

    # Reconstruct CDFs
    pred_t0_ns = torch.exp(pred_log_t0[batch_id:batch_id+1, :])
    target_t0_ns = target["t0_raw"][batch_id:batch_id+1, :].to(pred_coeffs.device)

    pred_cdf = cplib.reconstruct_cdf(pred_c, pred_t0_ns)
    target_cdf = cplib.reconstruct_cdf(target_c, target_t0_ns)

    tick_ns = cplib._tick_ns
    pred_cdf_pmt = pred_cdf[0, pmt_id, :]
    target_cdf_pmt = target_cdf[0, pmt_id, :]

    pred_pdf = cdf_to_pdf(pred_cdf_pmt, tick_ns)
    target_pdf = cdf_to_pdf(target_cdf_pmt, tick_ns)

    t_window = torch.arange(0, pred_pdf.shape[-1]) * tick_ns

    # t0 in ns for plotting
    pred_t0_val = pred_t0_ns[0, pmt_id]
    target_t0_val = target_t0_ns[0, pmt_id].to(pred_t0_val.device)
    t0s = torch.stack([target_t0_val, pred_t0_val], dim=-1)

    output = {
        "x_value": t_window,
        "visibility": torch.stack([target_v_linear, pred_v_linear], dim=-1),
        "pdf": torch.stack([target_pdf, pred_pdf], dim=-1),
        "cdf": torch.stack([target_cdf_pmt, pred_cdf_pmt], dim=-1),
        "t0": t0s,
        "position": x[batch_id] if x.dim() > 1 else x,
    }

    net_module.unfreeze_all()
    return output


def train_pca(cfg: dict):
    """Training loop for PCA-compressed photon library mode."""
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

    # Model
    net = SirenTV(cfg).to(DEVICE)
    net = torch.compile(net)
    if is_distributed:
        net = create_ddp_model(net, device_ids=[local_rank])

    # Data
    dl = create_compressed_dataloader(cfg, rank=rank, world_size=world_size)

    # fixed voxel for consistent WandB plots across iterations
    _plot_sample = dl.dataset[0]
    plot_x = _plot_sample["position"].unsqueeze(0).to(DEVICE)
    plot_target = {k: v.unsqueeze(0).to(DEVICE) for k, v in _plot_sample["target"].items()}
    if rank == 0:
        print(f"[train_pca] fixed plot voxel position: {plot_x.squeeze().cpu().tolist()}")

    # Compressed PLib for CDF reconstruction during logging
    cplib_cfg = cfg["compressed_plib"]
    cplib = CompressedPLib.load(
        cplib_cfg["filepath"],
        lazy=False,
        n_components=cplib_cfg.get("n_components"),
    )

    normalize_coeffs = bool(cplib_cfg.get("normalize_coeffs", False))
    if normalize_coeffs:
        cplib.compute_coeff_stats()

    # optional per-component EVR weighting for the coeffs loss
    evr_weight_coeffs = None
    if cplib_cfg.get("evr_weight_coeffs", False) and cplib.explained_variance_ratio is not None:
        K = cplib._n_components
        evr = cplib.explained_variance_ratio[:K].copy()
        evr = evr / evr.sum()
        evr_weight_coeffs = torch.from_numpy(evr.astype(np.float32)).to(DEVICE)
        if rank == 0:
            print(f"[train_pca] EVR weighting enabled: top-5 weights = {evr[:5]}")

    # Optimizer
    opt, sch, epoch = optimizer_factory(
        list(p for p in net.parameters() if p.requires_grad), cfg
    )
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        if rank == 0:
            print(f"[train_pca] resuming from iteration {iteration_ctr}, epoch {epoch_ctr}")

    # Losses, logger
    loss_fns = build_losses(cfg)
    regularizer = build_regularizer(cfg)
    logger = build_logger(cfg, net, rank=rank, world_size=world_size, is_distributed=is_distributed)

    if rank == 0:
        with open(os.path.join(logger.logdir, "train_cfg.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    # Training params
    train_cfg = cfg.get("train", {})
    epoch_max = train_cfg.get("max_epochs", int(1e20))
    iteration_max = train_cfg.get("max_iterations", int(1e20))
    save_every_iterations = train_cfg.get("save_every_iterations", -1)
    save_every_epochs = train_cfg.get("save_every_epochs", -1)
    reduction = train_cfg.get("reduction", "mean")

    amp = train_cfg.get("amp", False)
    if amp:
        scaler = torch.amp.GradScaler("cuda")

    weight_cfg = cfg.get("data", {}).get("weight", {})

    if rank == 0:
        print(f"[train_pca] max iterations {iteration_max}, max epochs {epoch_max}")
        print(f"[train_pca] distributed: {is_distributed}, world_size: {world_size}")

    stop_training = False
    losses = [float("inf")] * len(loss_fns)

    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        if is_distributed and hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
            dl.sampler.set_epoch(epoch_ctr)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        data_loading_start = time.time()

        pbar = tqdm(dl, desc=f"Epoch {epoch_ctr:<3d}", disable=(rank != 0))
        for batch_idx, data in enumerate(pbar):
            iteration_ctr += 1

            with (torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if amp else nullcontext()):
                x = data["position"].to(DEVICE, non_blocking=True)
                target = {
                    "v": data["target"]["v"].to(DEVICE, non_blocking=True),
                    "t0": data["target"]["t0"].to(DEVICE, non_blocking=True),
                    "coeffs": data["target"]["coeffs"].to(DEVICE, non_blocking=True),
                }

                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                data_loading_time = time.time() - data_loading_start

                # Forward pass
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                forward_start = time.time()
                pred = net(x)

                # Remap model output: model's "t" key -> "coeffs"
                pred["coeffs"] = pred.pop("t")

                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                forward_time = time.time() - forward_start

                # Weights
                weights = {
                    k: get_weight_by_vis(
                        target[k],
                        factor=weight_cfg[k].get("factor", None),
                        threshold=weight_cfg[k].get("threshold", 1e-8),
                    )
                    if (k in weight_cfg and weight_cfg[k].get("enable", False))
                    else 1.0
                    for k in target.keys()
                }

                if evr_weight_coeffs is not None:
                    shape = pred["coeffs"].shape  # (B, N_pmt, K)
                    w = evr_weight_coeffs.expand(shape)
                    if isinstance(weights.get("coeffs"), torch.Tensor):
                        weights["coeffs"] = weights["coeffs"] * w
                    else:
                        weights["coeffs"] = w

                # Loss
                losses_dict = compute_loss(pred, target, loss_fns, weights)
                keys, losses_vals = zip(*losses_dict.items())
                losses_vals = torch.stack(losses_vals)

                if reduction == "mean":
                    loss = torch.mean(losses_vals)
                elif reduction == "geometric_mean":
                    loss = torch.exp(torch.mean(torch.log(losses_vals)))
                elif reduction == "sum":
                    loss = torch.sum(losses_vals)
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")

                if regularizer is not None:
                    loss += regularizer(net)

                opt.zero_grad()
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                backward_start = time.time()

                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                if torch.cuda.is_available():
                    # torch.cuda.synchronize()
                    peak_mem_forward = torch.cuda.max_memory_allocated() / (1024**3)
                    torch.cuda.reset_peak_memory_stats()
                else:
                    peak_mem_forward = 0.0
                backward_time = time.time() - backward_start

            losses = losses_vals.detach().cpu().tolist()
            pbar.set_postfix_str(", ".join(f"{k}={v:.2e}" for k, v in zip(keys, losses)))

            if rank == 0 and iteration_ctr % 10 == 0:
                print(
                    f"[iter {iteration_ctr:>6d}] "
                    f"loss: {', '.join(f'{k}={v:.2e}' for k,v in zip(keys, losses))} | "
                    f"data: {data_loading_time:.3f}s | fwd: {forward_time:.3f}s | bwd: {backward_time:.3f}s | "
                    f"mem: {peak_mem_forward:.2f}GB"
                )

            if rank == 0:
                current_lr = opt.param_groups[0]["lr"]
                logger.record(
                    ["iter", "epoch", "lr", "data_loading_time"]
                    + [f"loss_{k}" for k in keys]
                    + ["loss"],
                    [iteration_ctr, epoch_ctr, current_lr, data_loading_time]
                    + losses
                    + [loss.item()],
                )

            if isinstance(logger, WandbLogger):
                per_rank_metrics = {
                    "gpu_memory_gb": float(torch.cuda.memory_allocated(local_rank) / (1024**3))
                    if torch.cuda.is_available()
                    else 0,
                    "loss": loss.detach().item(),
                }
                logger.log_per_rank_metrics(iteration_ctr, per_rank_metrics)

                # CDF/PDF reconstruction for fixed voxel
                inferred = _infer_pca_plot(net, plot_x, plot_target, cplib, denorm=normalize_coeffs)
                logger.plot(iteration_ctr, inferred)

                with torch.no_grad():
                    net_module = net.module if is_distributed else net
                    if hasattr(net_module, '_orig_mod'):
                        net_module = net_module._orig_mod
                    pred['v_linear'] = net_module._inv_xform_vis(pred['v'])
                    target['v_linear'] = net_module._inv_xform_vis(target['v'])
                    logger.step(iteration_ctr, target, pred)

                logger.commit(iteration_ctr)

            # Save checkpoint
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

            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            data_loading_start = time.time()

        if stop_training:
            break

        if sch is not None:
            sch.step(loss)

        epoch_ctr += 1

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
        print(f"[train_pca] Stopped at iteration {iteration_ctr}, epoch {epoch_ctr}")
        logger.write()
        logger.close()

    cplib.close()

    if is_distributed:
        dist.barrier()
