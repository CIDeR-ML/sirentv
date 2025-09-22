import torch
import numpy as np

import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        Args:
            t: tensor of shape [..., 1]
        Returns:
            embedding: tensor of shape [..., dim]
        """
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=device) * (np.log(self.max_period) / half_dim)
        )
        args = t * freqs.view(1, -1)
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2 == 1:  # if dim is odd, pad with one extra zero
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding
