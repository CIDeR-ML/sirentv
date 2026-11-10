import torch


def cdf_to_pdf(cdf: torch.Tensor, tick_size: float):
    return torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1])) / tick_size
