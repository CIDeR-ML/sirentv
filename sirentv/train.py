import os
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm
from slar.optimizers import optimizer_factory
from slar.utils import CSVLogger, get_device

from .io import PLibDataLoader
from .analysis import get_pred_target, log_pred_target
from .sirentv import SirenTV
from .utils import CSVLogger, WandbLogger
from . import utils

import wandb


def wandbcfg_to_dict(cfg: wandb.config, out_cfg: dict = None):
    if out_cfg is None:
        out_cfg = {}
    for k,v in cfg.items():
        layers = k.split(".")
        curr_cfg = out_cfg
        for layer in layers[:-1]:
            if layer not in curr_cfg:
                curr_cfg[layer] = {}
            curr_cfg = curr_cfg[layer]
        curr_cfg[layers[-1]] = v
    return out_cfg


def fix_cfg(cfg: wandb.config):
    cfg_dict = wandbcfg_to_dict(cfg)
    if 'hidden_features' in cfg_dict['model']['network']:
        return cfg_dict

    hidden_features = []
    hidden_layers = []
    for i in range(3):
        hidden_features.append(cfg_dict["model"]["network"].pop("hidden_features" + str(i)))
        hidden_layers.append(cfg_dict["model"]["network"].pop("hidden_layers" + str(i)))
    cfg_dict["model"]["network"]["hidden_features"] = hidden_features
    cfg_dict["model"]["network"]["hidden_layers"] = hidden_layers

    return cfg_dict

def train(cfg: str = None):
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
    if cfg is None:
        cfg = yaml.safe_load(
            open(os.path.join(os.path.dirname(__file__), "../templates/bvis-4848.yaml"))
        )

    # Initialize wandb
    wandb.init()
    if hasattr(wandb.config, 'model'):
        cfg = wandbcfg_to_dict(wandb.config, cfg)
        cfg = fix_cfg(cfg)

    # Set up device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("device"):
        DEVICE = get_device(cfg["device"]["type"])
    print('[train] using device', DEVICE)

    iteration_ctr = 0
    epoch_ctr = 0

    # Create necessary pieces: the model, optimizer, loss, logger.

    # Init model
    net = SirenTV(cfg).to(DEVICE)
    # Init data loader
    dl = PLibDataLoader(cfg, device=DEVICE)

    # Init optimizer, resuming if needed
    opt, sch, epoch = optimizer_factory(net.parameters(), cfg)
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        print(
            "[train] resuming training from iteration",
            iteration_ctr,
            "epoch",
            epoch_ctr,
        )

    # Set up loss functions. Must be defined in sirentv/utils.py
    loss_fns_str = (
        cfg.get("train", dict())
        .get("loss_fn", {})
        .get("functions", ["WeightedL2Loss"] * net.n_outs)
    )
    loss_fns = [getattr(utils, loss_fn)() for loss_fn in loss_fns_str]

    # Read in per-output loss function weights
    loss_fn_weights = (
        cfg.get("train", dict()).get("loss_fn", {}).get("weights", [1.0] * net.n_outs)
    )

    # Set up regularization if needed
    regularization_dict = cfg.get("train", dict()).get("regularization", dict())
    regularizer = None
    if regularization_dict:
        weight_decay = regularization_dict.get("weight_decay", 0)
        if regularization_dict.get("type", "L1") == 'L1':
            regularizer = utils.L1Regularization(weight_decay)
        elif regularization_dict.get("type", "L1") == 'L2':
            regularizer = utils.L2Regularization(weight_decay)

    # Set up logger
    logger_type = cfg.get("logger", dict()).get("type", "csv")
    logger = CSVLogger(cfg) if logger_type == "csv" else WandbLogger(cfg)
    logdir = logger.logdir
    if hasattr(logger, 'watch_grad'):
        logger.watch_grad(net)

    # Set the control parameters for the training loop
    train_cfg = cfg.get("train", dict())
    epoch_max = train_cfg.get("max_epochs", int(1e20))
    iteration_max = train_cfg.get("max_iterations", int(1e20))
    save_every_iterations = train_cfg.get("save_every_iterations", -1)
    save_every_epochs = train_cfg.get("save_every_epochs", -1)
    print(f"[train] train for max iterations {iteration_max} or max epochs {epoch_max}")

    # Store configuration   
    with open(os.path.join(logdir, "train_cfg.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # Start the training loop
    t0 = time.time()
    twait = time.time()
    stop_training = False
    losses = [np.inf, np.inf]
    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        for batch_idx, data in enumerate(
            tqdm(
                dl,
                desc="Epoch %-3d; Loss %-3s"
                % (epoch_ctr, ",".join(["%.2e" % loss for loss in losses])),
            )
        ):
            iteration_ctr += 1

            # Input data prep
            x = data["position"].contiguous()  # .to(DEVICE)
            weights = data["weight"].contiguous().squeeze()  # .to(DEVICE)
            target = data["target"].contiguous().squeeze()  # .to(DEVICE)
            target_linear = data["value"].contiguous().squeeze()  # .to(DEVICE)

            twait = time.time() - twait
            # Running the model, compute the loss, back-prop gradients to optimize.
            ttrain = time.time()
            pred = net(x)

            loss = 0
            losses = []
            feature_ctr = 0

            # Compute loss for each output type (visibility + time)
            for idx, n_features in enumerate(net.out_features):
                curr_loss = loss_fn_weights[idx] * loss_fns[idx](
                    pred[:, feature_ctr : feature_ctr + n_features],
                    target[:, feature_ctr : feature_ctr + n_features],
                    weights[:, feature_ctr : feature_ctr + n_features],
                )
                losses.append(curr_loss)
                feature_ctr += n_features
            losses = torch.stack(losses)

            # Combine visibility + time losses
            reduction = cfg.get("train", dict()).get('loss_fn', dict()).get("reduction", "mean")
            if reduction == "mean":
                loss = torch.mean(losses)
            elif reduction == "geometric_mean":
                loss = torch.exp(torch.mean(torch.log(losses)))
            else:
                raise ValueError(f"Unknown reduction method: {reduction}")

            # Add regularization on top
            if regularizer:
                loss += regularizer(net)

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
            ttrain = time.time() - ttrain

            # Log training parameters           
            logger.record(
                ["iter", "epoch", "ttrain", "twait"] + [f'loss_{i}' for i in range(len(losses))] + ["loss"],
                [iteration_ctr, epoch_ctr, ttrain, twait] + losses.detach().cpu().tolist() + [loss.item()],
            )

            # Step the logger
            pred_linear = dl.inv_xform_vis(pred)
            logger.step(iteration_ctr, target_linear, pred_linear)
            twait = time.time()

            # Save the model parameters if the condition is met
            if save_every_iterations > 0 and iteration_ctr % save_every_iterations == 0:
                filename = os.path.join(
                    logdir,
                    "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr),
                )

                net.save_state(filename, opt, sch, iteration_ctr)

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
                logdir, "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr)
            )
            net.save_state(filename, opt, sch, iteration_ctr / len(dl))
            logger.save(filename)

    print("[train] Stopped training at iteration", iteration_ctr, "epochs", epoch_ctr)
    logger.write()
    pred, target = get_pred_target(dl, net)
    log_pred_target(pred[:,48:], target[:,48:], name="timing_comparison")
    log_pred_target(pred[:,:48], target[:,:48], name="visibility_comparison")

    logger.close()


def main(sweep_id):
    wandb.agent(sweep_id, train, count=1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        # Train on specified configuration
        train(sys.argv[1])
    else:
        train()
#     sweep_config = {
#         "method": "bayes",
#         "metric": {"name": "loss", "goal": "minimize"},
#         "early_terminate": {"type": "hyperband", "min_iter": 100},
#         "parameters": {
#             "train.optimizer_param.lr": {
#                 "max": 1e-5,
#                 "min": 2.5e-7,
#                 "distribution": "log_uniform_values",
#             },
#             "model.network.hidden_features0": {
#                 "values": [128, 256, 512, 1024, 2048, 4096, 8192]
#             },
#             "model.network.hidden_features1": {
#                 "values": [128, 256, 512, 1024, 2048, 4096, 8192]
#             },
#             "model.network.hidden_features2": {
#                 "values": [128, 256, 512, 1024, 2048, 4096, 8192]
#             },
#             "model.network.hidden_layers0": {
#                 "min": 1,
#                 "max": 5,
#                 "distribution": "int_uniform",
#             },
#             "model.network.hidden_layers1": {
#                 "min": 1,
#                 "max": 5,
#                 "distribution": "int_uniform",
#             },
#             "model.network.hidden_layers2": {
#                 "min": 1,
#                 "max": 5,
#                 "distribution": "int_uniform",
#             },
#             "data.loader.batch_size": {"values": [4096, 8192, 16384]},
#         },
#     }

#     sweep_id = wandb.sweep(sweep_config, project="siren-2x2")
#     main(sweep_id)