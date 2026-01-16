from sirentv.weighting.builder import WEIGHTINGS
import torch
import torch.nn as nn

@WEIGHTINGS.register_module()
class ExponentialTimingWeighting(nn.Module):
    """
    Weight by timing bin with constant + exponential decay profile.
    weight[i] = const + A * exp(-P * i)
    where P = -log(const/A) / n_ticks and A is the peak value of the exponential at the start.

    Args
    ----
    n_ticks : int
        Number of timing bins (default 1000). The exponential decay will be 0 at the last tick.
    const : float
        Constant weight applied to all ticks (default 1.0)
    exp_peak : float or None
        Peak value of exponential at tick 0. If None, defaults to 0.1 * const
    """

    def __init__(self, const=1.0, exp_peak=None, exp_const=1e-3):
        super().__init__()
        self.const = const
        self.exp_peak = exp_peak
        self.exp_const = exp_const

    def forward(self, t):
        """
        Apply timing weighting.

        Parameters
        ----------
        t : torch.Tensor
            Timing tensor of shape (B, N_pmt, N_time) or similar

        Returns
        -------
        w : torch.Tensor
            Weight tensor with w.shape == t.shape
        """
        device = t.device
        n_time = t.shape[-1]

        A = self.exp_peak
        P = self.exp_const

        ticks = torch.arange(n_time, device=device, dtype=t.dtype)
        w_1d = self.const + A * torch.exp(-P * ticks)

        # expand to match input shape
        w = w_1d.expand(t.shape)

        return w


    def __repr__(self):
        return f"ExponentialTimingWeighting(const={self.const}, exp_peak={self.exp_peak}, exp_const={self.exp_const})"
