import copy

from sirentv.utils.registry import Registry

MODELS = Registry("models")

def build_model(cfg):
    """Build models."""
    return MODELS.build(copy.deepcopy(cfg))
