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
from sirentv.utils.comm import create_ddp_model
from sirentv.utils.log import CSVLogger, WandbLogger, Logger
from sirentv.utils.transform import pdf_to_cdf, cdf_to_pdf

def infer_single_pos_single_pmt(net: nn.Module, input_x: tensor, target: dict, tick_size: float, batch_id:long=0, pmt_id:long=40):
    """
    Function to get the visibility and pdf from any position and compare to target

    Parameters
    ----------
    net
    input_x
    target
    tick_size

    Returns
    -------

    """
    was_training = net.training
    net.eval()
    use_CDF = net._use_cdf

    # get a valid position based on true mask
    if 'v_mask' in target.keys():
        true_mask = target['v_mask'].squeeze()
        valid_indices = torch.where(true_mask)[0]
        if len(valid_indices) > 0:
            batch_id = valid_indices[0].item()  # ← Take FIRST valid index as scalar

    with torch.no_grad():
        pred: dict[str, torch.Tensor] = net(input_x)
    pred_v_linear = net._inv_xform_vis(pred["v"][batch_id, :])
    target_v_linear = target["v_linear"][batch_id, :].to(pred_v_linear.device)

    t0s = None
    if "t0" in target.keys():
        pred_t0 = pred['t0'][batch_id, pmt_id] * tick_size
        target_t0 = target['t0'][batch_id, pmt_id].to(pred_t0.device)
        t0s = torch.stack([target_t0, pred_t0], dim=-1)

    if use_CDF:
        pred_t_cdf = pred["t"][batch_id, pmt_id, :]
        target_t_cdf = target['t_linear'][batch_id, pmt_id, :].to(pred_t_cdf.device)

        pred_t_pdf = cdf_to_pdf(pred_t_cdf, tick_size)
        target_t_pdf = cdf_to_pdf(target_t_cdf, tick_size)
    else:
        pred_t_pdf = pred["t"][batch_id, pmt_id, :]
        target_t_pdf = target['t_linear'][batch_id, pmt_id, :].to(pred_t_pdf.device)

        pred_t_cdf = pdf_to_cdf(pred_t_pdf)
        target_t_cdf = pdf_to_cdf(target_t_pdf)

    t_window = torch.arange(0, pred_t_pdf.shape[-1])*tick_size

    output = {
        "x_value": t_window,
        "visibility": torch.stack([target_v_linear, pred_v_linear],dim=-1),
        "pdf": torch.stack([target_t_pdf, pred_t_pdf], dim=-1),
        "cdf": torch.stack([target_t_cdf, pred_t_cdf], dim=-1),
        "t0": t0s,
        "position": input_x[batch_id] if input_x.dim() > 1 else input_x
    }

    if was_training:
        net.train()
    return output