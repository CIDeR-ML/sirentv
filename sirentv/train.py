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

                x = data["position"].contiguous().to(DEVICE)
                weights = data["weight"].contiguous().squeeze().to(DEVICE)
                target = data["target"].contiguous().squeeze().to(DEVICE)
                target_linear = data["value"].contiguous().squeeze().to(DEVICE)

                twait = time.time() - twait
                # Running the model, compute the loss, back-prop gradients to optimize.
                ttrain = time.time()
                pred = net(x)
                # compute linear-domain prediction once for losses that need it
                pred_linear = dl.inv_xform_vis(pred)

                losses = compute_loss(
                    pred={"transformed": pred, "linear":pred_linear},
                    target={"transformed": target, "linear":target_linear},
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
    log_pred_target(pred[:,48:], target[:,48:], name="timing_comparison")
    log_pred_target(pred[:,:48], target[:,:48], name="visibility_comparison")

    if hasattr(net, "log_sigmas"):
        log_line(torch.exp(net.log_sigmas[0]).detach().cpu().numpy(), name=f"log_sigma_999")
        log_imshow(torch.exp(net.log_sigmas[1]).reshape(100,48).detach().cpu().numpy(), name=f"log_sigma_1_vis")
        log_line(torch.exp(net.log_sigmas[1]).detach().cpu().numpy(), name=f"log_sigma_1_linear")

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