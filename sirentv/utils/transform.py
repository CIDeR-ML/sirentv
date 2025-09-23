import torch
import torch.nn.functional as F

def pdf_to_cdf(pdf: torch.Tensor):
    return F.normalize(pdf, p=1, dim=-1).cumsum(-1)

def cdf_to_pdf(cdf: torch.Tensor, tick_size: float):
    return torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1])) / tick_size