from sirentv.weighting.builder import WEIGHTINGS
import torch
import torch.nn as nn

@WEIGHTINGS.register_module()
class ConstVisibilityWeighting(nn.Module):
    """
    Weight by visibility, `weight  = vis * factor`.
    Weights (after applying factor) below `threshold` are set to 1.

    Args
    ----
    factor : float or None
        Scaling factor applied to the visibility values.
        If None, it is set to `1 / max(vis)`.

    threshold : float
        Values below this threshold are set to 1.
    """

    def __init__(self, factor=None, threshold=1e-8):
        super().__init__()
        self.factor = factor
        self.threshold = threshold

    def forward(self, vis):
        """
        Apply weighting to the visibility tensor.

        Parameters
        ----------
        vis: torch.Tensor
            Visibility values.

        Returns
        -------
        w: torch.Tensor
            Weight values with `w.shape == vis.shape`.
        """
        factor = self.factor
        if factor is None:
            factor = 1 / torch.max(vis.clamp(min=1e-8))
        w = vis * factor
        w = w.clone()
        w[w < self.threshold] = 1.0
        return w
