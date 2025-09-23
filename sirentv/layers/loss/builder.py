import copy

from sirentv.utils.registry import Registry

LOSSES = Registry("losses")
REGULARIZERS = Registry("regularizers")

def build_loss(cfg):
    """Build losses."""
    return LOSSES.build(copy.deepcopy(cfg))

def build_regularizer(cfg):
    """Build regularizers."""
    return REGULARIZERS.build(copy.deepcopy(cfg))