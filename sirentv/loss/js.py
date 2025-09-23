import torch
import torch.nn as nn
import torch.nn.functional as F


class JSDivergenceLoss(nn.Module):
    def __init__(self, reduce_method="mean"):
        super().__init__()
        self.reduce = reduce_method

    def forward(self, pred, target, eps=1.0e-10):
        batch_size = pred.size(0)
        n_pmts = pred.size(1)
        n_tbins = pred.size(2)

        pred_clamped = torch.clamp(pred, min=eps)
        P = (target / (target.sum(dim=-1, keepdim=True) + eps)).view(-1, n_tbins)
        log_Q = F.log_softmax(pred_clamped, dim=-1)
        Q = log_Q.exp().view(-1, n_tbins)
        M = 0.5 * (P + Q)

        kl_pm = F.kl_div(M.log(), P, reduction="none")
        kl_qm = F.kl_div(M.log(), Q, reduction="none")
        js = 0.5 * (kl_pm + kl_qm)

        if self.reduce == "batchmean":
            return js.sum() / batch_size
        elif self.reduce == "mean":
            return js.sum() / (batch_size * n_pmts)
        elif self.reduce == "sum":
            return js.sum()
        elif self.reduce == "none":
            return js
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduce}")
