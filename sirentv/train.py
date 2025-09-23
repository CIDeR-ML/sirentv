import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sirentv.data.io import PLibDataLoader
from slar.optimizers import optimizer_factory
from slar.utils import get_device
from tqdm import tqdm

import sirentv.loss as losses
from sirentv.analysis import get_pred_target, log_imshow, log_line, log_pred_target

from sirentv.models import SirenTV
from sirentv.utils.comm import create_ddp_model
from sirentv.utils.log import CSVLogger, WandbLogger


def build_loss_fns(cfg, num_outputs):
    loss_fns = cfg.get("train", dict()).get("loss_fn", dict()).get("functions", ["WeightedL2Loss"]*num_outputs)
    loss_fns = [getattr(losses, fn)() for fn in loss_fns]
    loss_weights = cfg.get("train", dict()).get(
    "loss_fn", {}).get("weights", [1.0]*num_outputs)
    do_uncertainty = cfg.get("train", dict()).get(
        "loss_fn", {}).get("do_uncertainty", [False]*num_outputs)
    return loss_fns, loss_weights, do_uncertainty

def build_regularizer(cfg):
    if not (regularizer := cfg.get("train", dict()).get("regularization", dict())):
        return None
    return getattr(losses, regularizer.get("type", "l1").upper() + "Regularization")(
        regularizer.get("weight_decay", 0)
    )

def build_logger(cfg, net):
    logger_type = cfg.get("logger", dict()).get("type", "csv")
    logger = CSVLogger(cfg) if logger_type == "csv" else WandbLogger(cfg)
    if hasattr(logger, "watch_grad"):
        logger.watch_grad(net)
    return logger


def compute_loss(pred, target, weights, loss_fns, loss_fn_weights, loss_fn_uncertainty, net):
    losses = []
    feature_ctr = 0
    for idx, features in enumerate(net.out_features):
        if loss_fn_uncertainty[idx]:
            log_sigma = net.log_sigmas[idx]
            key = "linear" if loss_fns[idx].requires_linear_domain else "transformed"
            curr_loss = loss_fn_weights[idx] * loss_fns[idx](
                pred[key][:, feature_ctr : feature_ctr + features],
                target[key][:, feature_ctr : feature_ctr + features],
                weights[:, feature_ctr : feature_ctr + features],
                log_sigma.unsqueeze(0),
            )
        else:
            lfn = loss_fns[idx]
            use_linear = bool(getattr(lfn, 'requires_linear_domain', False))
            key = "linear" if use_linear else "transformed"
            curr_loss = loss_fn_weights[idx] * lfn(
                pred[key][:, feature_ctr : feature_ctr + features],
                target[key][:, feature_ctr : feature_ctr + features],
                weights[:, feature_ctr : feature_ctr + features],
            )
        losses.append(curr_loss)
        feature_ctr += features
    losses = torch.stack(losses)
    return losses

def compute_tof(source_pos, pmt_coords, r_index=1.374):
    """
    Function to compute time of flight of photon traversing in the medium with given refractive index r_index
    Parameters
    ----------
    source_pos : tensor (N, 3)
    pmt_coords : tensor (N_pmt, 3)
    r_index : float

    Returns
    -------
    tof: tensor (N, N_pmt)
    """

    speed_c = 299.792 # mm/ns
    dist = torch.cdist(source_pos, pmt_coords, p=2)  # (N, N_pmt)
    tof = dist/speed_c*r_index
    return tof

def substract_tof(vis, tof):
    assert len(vis) == len(tof), "Visibility and tof tensors length mismatched."

    V, N, T = vis.shape
    t_idx = torch.arange(T, device=vis.device).expand(V, N, T)

    t_shift = (tof / 0.1).unsqueeze(-1).long().to(vis.device)  # hardcoded 100ps per bin for 100ns window
    vis_shifted = torch.zeros_like(vis)

    source_t_idx = t_idx + t_shift
    valid_mask = (source_t_idx >= 0) & (source_t_idx < T)  # Source must be within original tensor bounds

    v_coords = torch.arange(V, device=vis.device).view(V, 1, 1).expand(V, N, T)
    n_coords = torch.arange(N, device=vis.device).view(1, N, 1).expand(V, N, T)

    # Only copy where the source position is valid
    vis_shifted[valid_mask] = vis[v_coords[valid_mask], n_coords[valid_mask], source_t_idx[valid_mask]]
    """
    vis_argmax = vis.argmax(dim=2)  # Shape: [V, N]
    vis_shifted_argmax = vis_shifted.argmax(dim=2)  # Shape: [V, N]

    # Calculate actual shift (difference in argmax positions)
    actual_shift = vis_argmax - vis_shifted_argmax  # Should equal expected_shift
    # Check if shifts match (accounting for cases where peak might be clipped)
    shift_matches = (actual_shift == t_shift.squeeze(-1))

    print(f"Original argmax positions (first 2x2): {vis_argmax[:2, :2]}")
    print(f"Shifted argmax positions (first 2x2): {vis_shifted_argmax[:2, :2]}")
    print(f"Expected shift (first 2x2): {t_shift[:2, :2]}")
    print(f"Actual shift (first 2x2): {actual_shift[:2, :2]}")
    print(f"Shifts match (first 2x2): {shift_matches[:2, :2]}")

    # Summary statistics
    print(f"Percentage of shifts that match exactly: {shift_matches.float().mean().item() * 100:.1f}%")
    """
    vis_shifted = vis_shifted.view(V, -1)

    return vis_shifted

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

    correct_tof = cfg['train'].get('correct_tof', False)
    pmt_coord_file = cfg['data']['geometry'].get('pmt_coords', None)
    detector_yrange = cfg['data']['geometry'].get('detector_y')
    detector_zrange = cfg['data']['geometry'].get('detector_z')
    assert len(detector_yrange)==2 and len(detector_zrange)==2, "Detector ranges not input correctly"
    detector_ynorm = abs(yrange[1]-yrange[0])
    detector_znorm = abs(zrange[1]-zrange[0])

    assert pmt_coord_file is not None and os.path.isfile(pmt_coord_file), "PMT coord file is not loaded correctly"
    pmt_coords = torch.tensor(np.loadtxt(pmt_coord_file, delimiter=','), dtype=torch.float32).to(DEVICE)

    num_outputs = net.n_outs
    reduction = cfg.get("train", dict()).get('loss_fn', dict()).get("reduction", "mean")
    if reduction == "weighted_mean":
        init_logits = cfg.get("train", dict()).get('loss_fn', dict()).get("weighted_mean_logits", [0.0, 1.5])
        reduction_weights = torch.tensor(init_logits, requires_grad=True, device=DEVICE)
    else:
        reduction_weights = None

    net.to(DEVICE)
    opt, sch, epoch = optimizer_factory(list(p for p in net.parameters() if p.requires_grad) + ([reduction_weights] if reduction_weights is not None else []), cfg)
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        print(
            "[train] resuming training from iteration",
            iteration_ctr,
            "epoch",
            epoch_ctr,
        )

    loss_fns, loss_fn_weights, loss_fn_uncertainty = build_loss_fns(cfg, num_outputs)
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


    # amp flag
    amp = train_cfg.get("amp", False)
    if amp:
        scaler = torch.amp.GradScaler('cuda')
    print(f"[train] train for max iterations {iteration_max} or max epochs {epoch_max}")


    # Start the training loop
    twait = time.time()
    stop_training = False
    losses = [np.inf, np.inf]

    # through epochs
    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        # through batches
        for batch_idx, data in enumerate(tqdm(dl, desc="Epoch %-3d; Loss %-3s" % (epoch_ctr, ",".join(["%.2e" % l for l in losses])))):
            iteration_ctr += 1
            with (torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if amp else nullcontext()):

                pmt_coords_norm = torch.cat([pmt_coords[:, 0], pmt_coords[:, 1]/detector_ynorm, pmt_coords[:, 2]/detector_znorm], dim=1).to(DEVICE)
                x = data["norm_position"].contiguous().to(DEVICE)
                pmt_coords_exp = torch.tile(pmt_coord_norm.unsqueeze(0), (x.shape[0], 1, 1))
                x_exp = torch.tile(x.unsqueeze(1), (1, pmt_coords_exp.shape[1], 1))
                x_input = torch.cat([x_exp, pmt_coords_exp], dim=-1).contiguous().to(DEVICE)
                
                raw_position = data["raw_position"].contiguous().to(DEVICE)
                weights = data["weight"].contiguous().squeeze().to(DEVICE)
                target = data["target"].contiguous().squeeze().to(DEVICE)
                target_linear = data["value"].contiguous().squeeze().to(DEVICE)

                if (correct_tof):
                    tof = compute_tof(raw_position, pmt_coords)
                    target = substract_tof(target, tof)
                    target_linear = substract_tof(target_linear, tof)

                twait = time.time() - twait
                # Running the model, compute the loss, back-prop gradients to optimize.
                ttrain = time.time()
                pred = net(x)
                # compute linear-domain prediction once for losses that need it
                pred_linear = dl.inv_xform_vis(pred)

                target_sum = torch.sum(target, dim=-1)
                target_for_loss = torch.cat([target_sum, target], dim=-1)
                target_linear_sum = torch.sum(target_linear, dim=-1)
                target_linear_for_loss = torch.cat([target_linear_sum, target_linear], dim=-1)

                losses = compute_loss(
                    pred={"transformed": pred, "linear":pred_linear},
                    target={"transformed": target_for_loss, "linear":target_linear_for_loss},
                    weights=weights,
                    loss_fns=loss_fns,
                    loss_fn_weights=loss_fn_weights,
                    loss_fn_uncertainty=loss_fn_uncertainty,
                    net=net,
                ) # (num_outputs,) <-- = 2

                if reduction == "mean":
                    loss = torch.mean(losses)
                elif reduction == "geometric_mean":
                    loss = torch.exp(torch.mean(torch.log(losses)))
                elif reduction == "weighted_mean":
                    loss_weight_logits = F.softmax(reduction_weights, dim=0)
                    loss = torch.mean(losses * loss_weight_logits)
                elif reduction == "sum":
                    loss = torch.sum(losses)
                else:
                    raise ValueError(f"Unknown reduction method: {reduction}")

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
            ttrain = time.time() - ttrain

            # get current learning rate
            current_lr = opt.param_groups[0]['lr']

            # Log training parameters           
            logger.record(
                ["iter", "epoch", "ttrain", "twait", "lr"] + [f'loss_{i}' for i in range(len(losses))] + ["loss"] + (['alpha', 'beta'] if 'loss_weight_logits' in locals() else []),
                [iteration_ctr, epoch_ctr, ttrain, twait, current_lr] + losses.detach().cpu().tolist() + [loss.item()] + (loss_weight_logits.detach().cpu().tolist() if 'loss_weight_logits' in locals() else []),
            )

            # Step the logger
            pred_linear = dl.inv_xform_vis(pred)
            logger.step(iteration_ctr, target_linear, pred_linear)
            twait = time.time()

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
    feature_ctr = 0
    for idx, features in enumerate(net.out_features):
        log_pred_target(pred[:,feature_ctr:feature_ctr+features], target[:,feature_ctr:feature_ctr+features], name=f"comparison_{idx}")

    # if hasattr(net, "log_sigmas"):
    #     log_line(torch.exp(net.log_sigmas[0]).detach().cpu().numpy(), name=f"log_sigma_999")
    #     log_imshow(torch.exp(net.log_sigmas[1]).reshape(100,48).detach().cpu().numpy(), name=f"log_sigma_1_vis")
    #     log_line(torch.exp(net.log_sigmas[1]).detach().cpu().numpy(), name=f"log_sigma_1_linear")

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