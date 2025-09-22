import numpy as np
# from sirentv.io import PhotonLibDataset, PLibDataLoader
from slar.siren_vis import SirenVis
import torch
import torch.nn.functional as F
import yaml
from slar.io import PLibDataLoader
from slar.optimizers import optimizer_factory
from slar.utils import get_device
from sirentv.utils.log import CSVLogger, WandbLogger
import utils
import sirentv as siren_models
import os
import time
from tqdm import tqdm
import wandb
from analysis import log_pred_target, get_pred_target, log_line, log_imshow
from contextlib import nullcontext


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
    model_name = cfg.get("model", dict()).get("name", "SirenTV")
    if not hasattr(siren_models, model_name):
        raise ValueError(f"Unknown model class '{model_name}' in sirentv.py")
    net = getattr(siren_models, model_name)(cfg)
    net.to(DEVICE)
    dl = PLibDataLoader(cfg, device='cpu')

    num_outputs = net.n_outs
    reduction = cfg.get("train", dict()).get('loss_fn', dict()).get("reduction", "mean")
    if reduction == "weighted_mean":
        init_logits = cfg.get("train", dict()).get('loss_fn', dict()).get("weighted_mean_logits", [0.0, 1.5])
        reduction_weights = torch.tensor(init_logits, requires_grad=True, device=DEVICE)
    else:
        reduction_weights = None

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

    # read in loss functions
    loss_fns = cfg.get("train", dict()).get(
        "loss_fn", {}).get("functions", ["WeightedL2Loss"]*num_outputs)
    # instantiate loss functions
    loss_fns = [getattr(utils, loss_fn)() for loss_fn in loss_fns]
    # read in loss function weights
    loss_fn_weights = cfg.get("train", dict()).get(
        "loss_fn", {}).get("weights", [1.0]*num_outputs)
    loss_fn_uncertainty = cfg.get("train", dict()).get(
        "loss_fn", {}).get("uncertainty", [False]*num_outputs)

    # read in regularization
    regularization = cfg.get("train", dict()).get("regularization", dict())
    # instantiate regularization
    regularizer = None
    if regularization:
        regularizer = getattr(
            utils, regularization.get("type", "l1").upper() + "Regularization"
        )(regularization.get("weight_decay", 0))


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
    amp = train_cfg.get("amp", False)
    if amp:
        scaler = torch.amp.GradScaler('cuda')
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
        for batch_idx, data in enumerate(tqdm(dl, desc="Epoch %-3d; Loss %-3s" % (epoch_ctr, ",".join(["%.2e" % loss for loss in losses])))):
            iteration_ctr += 1

            # Input data prep
            context = torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if amp else nullcontext()
            with context:
                x = data["position"].contiguous().to(DEVICE)
                weights = data["weight"].contiguous().squeeze().to(DEVICE)
                target = data["target"].contiguous().squeeze().to(DEVICE)
                target_linear = data["value"].contiguous().squeeze().to(DEVICE)

                twait = time.time() - twait
                # Running the model, compute the loss, back-prop gradients to optimize.
                ttrain = time.time()
                pred = net(x)

                # print('target', target[:,48:].min(), target[:,48:].max(), target[:,48:].reshape(target.shape[0], 48, 100).sum(-1))
                # print('pred', pred[:,48:].min(), pred[:,48:].max(), pred[:,48:].reshape(pred.shape[0], 48, 100).sum(-1))
                # compute linear-domain prediction once for losses that need it
                pred_linear = dl.inv_xform_vis(pred)

                loss = 0
                losses = []
                feature_ctr = 0
                for idx, features in enumerate(net.out_features):
                    if loss_fn_uncertainty[idx]:
                        log_sigma = net.log_sigmas[idx]
                        # uncertainty path assumes transformed-domain mse-like targets
                        curr_loss = loss_fn_weights[idx] * loss_fns[idx](
                            pred[:, feature_ctr : feature_ctr + features],
                            target[:, feature_ctr : feature_ctr + features],
                            weights[:, feature_ctr : feature_ctr + features],
                            log_sigma.unsqueeze(0),
                        )
                    else:
                        lfn = loss_fns[idx]
                        use_linear = bool(getattr(lfn, 'requires_linear_domain', False))
                        pred_slice = (
                            pred_linear[:, feature_ctr : feature_ctr + features]
                            if use_linear
                            else pred[:, feature_ctr : feature_ctr + features]
                        ) 
                        linear_slice = target_linear[:, feature_ctr : feature_ctr + features]
                        target_slice = (
                            linear_slice
                            if use_linear
                            else target[:, feature_ctr : feature_ctr + features]
                        )
                        curr_loss = loss_fn_weights[idx] * lfn(
                            pred_slice,
                            target_slice,
                            weights[:, feature_ctr : feature_ctr + features],
                        )
                    losses.append(curr_loss)
                    feature_ctr += features
                losses = torch.stack(losses)

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
            # logger.save(filename)


    print("[train] Stopped training at iteration", iteration_ctr, "epochs", epoch_ctr)
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