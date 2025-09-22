"""
Instant-NGP–style neural field for PMT waveform amplitudes

phi : (x, y, z, t, pmt_id [, optional pmt_static_feats]) -> A >= 0

- Fast multi-resolution hash-grid encoding over 4D coords (x,y,z,t)
- Per-PMT learned embedding, plus optional static PMT features (e.g., position, normal)
- Tiny MLP head outputs either amplitude via Softplus or log-amplitude directly
- Optional ToF-aware preprocessing (subtract direct c/n time and/or apply causal gate)

This file contains:
  * MultiResHashEncoder (4D hash grid with linear interpolation)
  * PMTNeuralField (end-to-end model)
  * Losses (Poisson deviance and log-MSE)
  * Minimal training sketch

No external deps beyond PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------
# Utilities
# -------------------------------

def _make_offsets(D: int):
    """All 2^D binary corner offsets for D-d interpolation, shape (2^D, D)."""
    combos = list(itertools.product([0, 1], repeat=D))
    return torch.tensor(combos, dtype=torch.long)  # (2^D, D)


def _primes_for_hash(D: int):
    """Return D large distinct 64-bit primes for spatial hashing."""
    # A set of well-spaced 64-bit-ish primes (fit in signed 64-bit when multiplied by small ints).
    primes = [1_140_380_659, 1_466_926_071, 1_867_466_593, 2_145_313_613, 2_327_235_169, 2_643_559_489, 2_954_089_357]
    assert D <= len(primes)
    return torch.tensor(primes[:D], dtype=torch.long)


def _fast_hash(coords: torch.Tensor, primes: torch.Tensor, hash_size: int) -> torch.Tensor:
    """Fast XOR-mix spatial hash.

    Args:
        coords: (..., D) int64 non-negative integer coordinates
        primes: (D,) int64 primes
        hash_size: int table size (modulus)
    Returns:
        idx: (...) int64 indices in [0, hash_size)
    """
    x = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
    for d in range(coords.shape[-1]):
        x ^= coords[..., d] * primes[d]
    # final avalanching & modulus
    x ^= (x >> 32)
    if hash_size & (hash_size - 1) == 0:
        # power-of-two size -> bitmask
        idx = x & (hash_size - 1)
    else:
        idx = x.remainder(hash_size)
    return idx


# -------------------------------
# Hash-grid encoder (4D)
# -------------------------------

class MultiResHashEncoder(nn.Module):
    """Multi-resolution hash-grid encoder for 4D inputs.

    Input x must be normalized to [0,1]^4 over (x,y,z,t).

    For each level ℓ, with resolution R_ℓ and table size T_ℓ, we do 4D linear
    interpolation from a hash-indexed feature table of shape (T_ℓ, F).

    Based on Instant-NGP (Müller et al., 2022), simplified and pure PyTorch.
    """

    def __init__(
        self,
        input_dim: int = 4,
        num_levels: int = 16,
        level_dim: int = 2,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
        hash_size: int = 2 ** 19,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert input_dim == 4, "This encoder is configured for 4D (x,y,z,t)."
        self.D = input_dim
        self.L = num_levels
        self.F = level_dim
        self.base = base_resolution
        self.growth = per_level_scale
        self.dtype = dtype
        self.register_buffer("primes", _primes_for_hash(self.D))
        self.register_buffer("offsets", _make_offsets(self.D))  # (16,4)

        # Per-level hash tables
        self.hash_size = hash_size
        tables = []
        for _ in range(self.L):
            # Small init helps stability when paired with Softplus head
            table = nn.Parameter(1e-4 * torch.randn(hash_size, self.F, dtype=dtype, device=device))
            tables.append(table)
        self.tables = nn.ParameterList(tables)

        # Precompute per-level resolutions (as floats for scaling)
        res = [int(math.floor(self.base * (self.growth ** l))) for l in range(self.L)]
        self.register_buffer("resolutions", torch.tensor(res, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 4D normalized coordinates.

        Args:
            x: (N, 4) in [0,1]
        Returns:
            feats: (N, L*F)
        """
        assert x.shape[-1] == self.D
        x = x.clamp(0.0, 1.0)
        N = x.shape[0]
        device = x.device

        feats_per_level = []
        for l in range(self.L):
            Rl = self.resolutions[l].item()
            table = self.tables[l]
            # scaled coordinates in grid units
            xf = x * Rl  # (N,4)
            x0 = torch.floor(xf).to(torch.long)  # (N,4)
            # allow wrapping at boundary for interpolation (periodic padding)
            # For PMT scenes, clamping is also fine; here we wrap to mimic Instant-NGP default.
            x1 = (x0 + 1)
            w = (xf - x0.to(xf.dtype))  # (N,4) fractional part in [0,1)

            # 16 corners
            # corner integer coords (N,16,4)
            corners = (x0[:, None, :] + self.offsets[None, :, :])
            # wrap into [0, Rl-1]
            corners = corners.remainder(Rl)

            # hash indices per corner (N,16)
            idx = _fast_hash(corners.reshape(-1, 4), self.primes, self.hash_size).reshape(N, -1)
            # fetch corner features (N,16,F)
            corner_feats = table[idx]

            # weights per corner (N,16)
            # For each corner c with bits b0..b3, weight is prod_j (w_j if b_j=1 else (1-w_j))
            # Vectorize by computing per-dim contributions and multiplying
            # weights_dim: list of (N,16) per dim
            weights = 1.0
            for d in range(4):
                wd = w[:, d:d+1]  # (N,1)
                bd = self.offsets[:, d].to(x.device).float()[None, :]  # (1,16)
                # choose w or (1-w) based on bit
                wd_full = torch.where(bd > 0.5, wd, 1.0 - wd)  # (N,16)
                weights = weights * wd_full

            # weighted sum over 16 corners -> (N,F)
            fe = (corner_feats * weights[..., None]).sum(dim=1)
            feats_per_level.append(fe)

        enc = torch.cat(feats_per_level, dim=-1)  # (N, L*F)
        return enc


# -------------------------------
# PMT Neural Field
# -------------------------------

@dataclass
class FieldConfig:
    # Hash encoder
    num_levels: int = 16
    level_dim: int = 2
    base_resolution: int = 16
    per_level_scale: float = 1.5
    hash_size: int = 2 ** 19
    # MLP
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "silu"  # or "relu"
    # Outputs
    output_mode: str = "amplitude"  # "amplitude" via Softplus or "log_amplitude" direct
    softplus_beta: float = 1.0
    # PMT embeddings / static features
    pmt_emb_dim: int = 8
    pmt_static_dim: int = 0  # set to 6 if you append (px,py,pz,nx,ny,nz), normalized
    # ToF options (optional; for pure baseline keep both False)
    use_tof_shift: bool = False
    refractive_index: float = 1.38  # LAr ~1.38 at 128 nm; update as needed
    causal_gate_sigma: Optional[float] = None  # e.g., 1.0 ns; None disables gate


class PMTNeuralField(nn.Module):
    """Instant-NGP–style neural field for PMT waveforms.

    Inputs expected:
      - coords: (N,4) torch.float32 normalized to [0,1] over scene box and time window
                coords[..., :3] = (x,y,z), coords[..., 3] = t
      - pmt_id: (N,) long tensor of PMT indices in [0, n_pmts)

    Optional:
      - pmt_static: (n_pmts, pmt_static_dim) static per-PMT features, pre-normalized
      - scene_box: ((3,), (3,)) min/max for undoing normalization (only if ToF is used)
      - pmt_positions: (n_pmts,3) physical positions (only if ToF is used)

    If cfg.use_tof_shift is True, we subtract t0 = ||x - p_i|| / (c/n) from t before encoding.
    If cfg.causal_gate_sigma is not None, we multiply outputs by sigmoid((t - t0)/sigma).
    """

    def __init__(self, n_pmts: int, cfg: FieldConfig, pmt_static: Optional[torch.Tensor] = None,
                 scene_box: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 pmt_positions: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.n_pmts = n_pmts

        self.encoder = MultiResHashEncoder(
            input_dim=4,
            num_levels=cfg.num_levels,
            level_dim=cfg.level_dim,
            base_resolution=cfg.base_resolution,
            per_level_scale=cfg.per_level_scale,
            hash_size=cfg.hash_size,
        )

        self.pmt_emb = nn.Embedding(n_pmts, cfg.pmt_emb_dim)
        self.register_buffer("pmt_static", pmt_static if pmt_static is not None else None)

        in_dim = cfg.num_levels * cfg.level_dim + cfg.pmt_emb_dim + (cfg.pmt_static_dim or 0)

        layers = []
        last = in_dim
        act = nn.SiLU() if cfg.activation == "silu" else nn.ReLU()
        for _ in range(cfg.num_layers):
            layers += [nn.Linear(last, cfg.hidden_dim), act]
            last = cfg.hidden_dim
        self.mlp = nn.Sequential(*layers)

        self.head = nn.Linear(last, 1)
        self.softplus = nn.Softplus(beta=cfg.softplus_beta)

        # For ToF/gating (optional)
        self.register_buffer("scene_min", None)
        self.register_buffer("scene_max", None)
        self.register_buffer("pmt_positions", None)
        if cfg.use_tof_shift:
            assert scene_box is not None and pmt_positions is not None, (
                "ToF shift requires scene_box=(min,max) and pmt_positions.")
            self.scene_min = scene_box[0].float()
            self.scene_max = scene_box[1].float()
            self.pmt_positions = pmt_positions.float()

    @staticmethod
    def _unnormalize_coords(coords01: torch.Tensor, scene_min: torch.Tensor, scene_max: torch.Tensor) -> torch.Tensor:
        return scene_min + coords01 * (scene_max - scene_min)

    def _compute_tof(self, coords01: torch.Tensor, pmt_id: torch.Tensor) -> torch.Tensor:
        # coords01: (N,4) in [0,1]; we only need xyz
        xyz = self._unnormalize_coords(coords01[..., :3], self.scene_min, self.scene_max)  # (N,3)
        p = self.pmt_positions[pmt_id]  # (N,3)
        d = torch.norm(p - xyz, dim=-1)  # meters (assuming scene_min/max in meters)
        c = 299.792458  # mm/ns
        v = c / self.cfg.refractive_index
        t0 = d / v  # seconds
        return t0

    def forward(self, coords01: torch.Tensor, pmt_id: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            coords01: (N,4) normalized coords in [0,1] for (x,y,z,t)
            pmt_id: (N,) long indices
        Returns:
            amplitude A: (N,1) >= 0  (if output_mode=="amplitude")
            or log-amplitude: (N,1) (if output_mode=="log_amplitude")
        """
        assert coords01.ndim == 2 and coords01.shape[1] == 4
        assert pmt_id.ndim == 1 and pmt_id.shape[0] == coords01.shape[0]

        coords = coords01
        if self.cfg.use_tof_shift:
            t0 = self._compute_tof(coords01, pmt_id)  # seconds
            # We assume time (coords[...,3]) was normalized from [t_min, t_max] seconds
            # To apply ToF subtraction in normalized space, we need the same scaling.
            # Here we approximate by linearizing w.r.t. the original [t_min, t_max]:
            # Let t_phys = t_min + coords[...,3]*(t_max-t_min). Then
            # (t_phys - t0) normalized back -> coords[...,3] - t0/(t_max-t_min)
            # The caller should make sure that t0/(t_max-t_min) is typically small enough.
            # For a pure baseline, disable use_tof_shift.
            # NOTE: This stays differentiable for parameters; t0 depends only on inputs.
            # If you want strict causality, also enable a causal gate via cfg.causal_gate_sigma.
            t_norm_shift = t0 / (1.0)  # Placeholder: set time window duration to 1 in normalized units
            coords = coords.clone()
            coords[..., 3] = coords[..., 3] - t_norm_shift.clamp_max(1.0)

        enc = self.encoder(coords)  # (N, L*F)
        feats = [enc, self.pmt_emb(pmt_id)]
        if self.cfg.pmt_static_dim and self.pmt_static is not None:
            feats.append(self.pmt_static[pmt_id])
        h = torch.cat(feats, dim=-1)
        h = self.mlp(h)
        out = self.head(h)

        if self.cfg.output_mode == "amplitude":
            if self.cfg.causal_gate_sigma is not None and self.cfg.use_tof_shift:
                # Apply causal gate in *physical* time. See note in _compute_tof for normalization.
                # Here we approximate by assuming coords01[...,3] is already proportional to time.
                t0 = self._compute_tof(coords01, pmt_id)
                gate = torch.sigmoid((coords01[..., 3] - t0) / (self.cfg.causal_gate_sigma))
                return self.softplus(out) * gate.unsqueeze(-1)
            return self.softplus(out)
        elif self.cfg.output_mode == "log_amplitude":
            return out
        else:
            raise ValueError(f"Unknown output_mode: {self.cfg.output_mode}")


# -------------------------------
# Losses & metrics
# -------------------------------

def poisson_deviance(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Generalized KL / Poisson deviance: pred, target >= 0 (use Softplus head)."""
    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)
    return (pred - target + target * (target.add(eps).log() - pred.log())).mean()


def log_mse(pred_log: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """MSE on log-amplitudes. Use with output_mode=="log_amplitude"."""
    return (pred_log - (target.add(eps).log())).pow(2).mean()


# -------------------------------
# Minimal training sketch
# -------------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 20_000
    batch_size: int = 131_072  # hash encoders love big batches
    loss_type: str = "poisson"  # or "logmse"
    amp: bool = True


class LUTDataset(torch.utils.data.Dataset):
    """Wraps precomputed LUT samples.

    You provide arrays/tensors:
      coords01: (N,4) normalized [0,1]
      pmt_id:   (N,) long
      amplitude:(N,) float >= 0
    """

    def __init__(self, coords01: torch.Tensor, pmt_id: torch.Tensor, amplitude: torch.Tensor):
        assert coords01.shape[0] == pmt_id.shape[0] == amplitude.shape[0]
        self.coords01 = coords01.float()
        self.pmt_id = pmt_id.long()
        self.amplitude = amplitude.float()

    def __len__(self):
        return self.coords01.shape[0]

    def __getitem__(self, idx):
        return self.coords01[idx], self.pmt_id[idx], self.amplitude[idx]


def train_one_field(model: PMTNeuralField, ds: LUTDataset, tcfg: TrainConfig, device: str = "cuda"):
    model = model.to(device)
    loader = torch.utils.data.DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(tcfg.amp and device.startswith("cuda")))

    loss_fn = poisson_deviance if tcfg.loss_type == "poisson" else log_mse

    model.train()
    step = 0
    while step < tcfg.steps:
        for coords01, pmt_id, amp in loader:
            coords01 = coords01.to(device)
            pmt_id = pmt_id.to(device)
            amp = amp.to(device).unsqueeze(-1)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(tcfg.amp and device.startswith("cuda"))):
                out = model(coords01, pmt_id)
                if tcfg.loss_type == "poisson":
                    loss = loss_fn(out, amp)
                else:
                    loss = loss_fn(out, amp)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % 100 == 0:
                print(f"step {step:>6d}  loss {loss.item():.5e}")
            step += 1
            if step >= tcfg.steps:
                break


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Synthetic tiny demo with random data (replace with your LUT samples)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_pmts = 81
    N = 1_000_000
    coords01 = torch.rand(N, 4)
    pmt_id = torch.randint(0, n_pmts, (N,))
    # Fake target: a few bumps in time + PMT id preference
    t = coords01[:, 3:4]
    target = (
        5.0 * torch.exp(-((t - 0.2) ** 2) / (2 * 0.01 ** 2))
        + 2.5 * torch.exp(-((t - 0.55) ** 2) / (2 * 0.02 ** 2))
        + 0.1
    )
    target *= 0.5 + (pmt_id.float().unsqueeze(-1) % 7) / 7.0
    target = target.squeeze(-1).clamp_min(0)

    ds = LUTDataset(coords01, pmt_id, target)

    cfg = FieldConfig(
        num_levels=16,
        level_dim=2,
        base_resolution=16,
        per_level_scale=1.5,
        hash_size=2 ** 18,
        hidden_dim=64,
        num_layers=2,
        activation="silu",
        output_mode="amplitude",  # use "log_amplitude" + log-MSE if your targets span many orders of magnitude
        pmt_emb_dim=8,
        pmt_static_dim=0,
        use_tof_shift=False,        # True if you wish to subtract analytic ToF (provide scene_box & pmt_positions)
        causal_gate_sigma=None,
    )

    model = PMTNeuralField(n_pmts=n_pmts, cfg=cfg)

    tcfg = TrainConfig(lr=1e-3, steps=2000, batch_size=32000, loss_type="poisson", amp=True)
    train_one_field(model, ds, tcfg, device=device)

    # After training, model(coords01, pmt_id) yields amplitudes in (0, inf).
