import copy

from sirentv.utils.registry import Registry

WEIGHTINGS = Registry("weightings")

def build_weighting(cfg):
    """Build weightings."""
    return WEIGHTINGS.build(copy.deepcopy(cfg))
