"""
Instant-NGP–style neural field for PMT waveform amplitudes (factorized 3D⊕1D) — 100 ns window, 0.1 ns resolution ready

phi : (x, y, z, t, pmt_id [, optional pmt_static_feats]) -> A >= 0

This version factorizes space/PMT from time so you can:
  * Encode (x,y,z) once per (voxel, PMT) and reuse across all T time ticks
  * Vectorize over a whole time vector t_vec in a single bilinear projection

Includes:
  - MultiResHashEncoderND (hash grid for D∈{1,3})
  - FourierTimeEncoder (fast 1D time features) + optional 1D hash encoder
  - PMTNeuralFieldFactorized with
      forward_sample(xyz01, t01, pmt_id) -> A
      forward_waveform(xyz01, pmt_id, t01_vec) -> A matrix (N, T)
  - Losses (Poisson deviance, log-MSE)
  - Minimal training & eval sketch

No external deps beyond PyTorch. Defaults below are set for a 100 ns window with 0.1 ns ticks (1000 samples).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Utilities
# -------------------------------

def _make_offsets(D: int):
    combos = list(itertools.product([0, 1], repeat=D))
    return torch.tensor(combos, dtype=torch.long)  # (2^D, D)


def _primes_for_hash(D: int):
    primes = [1_140_380_659, 1_466_926_071, 1_867_466_593, 2_145_313_613, 2_327_235_169, 2_643_559_489, 2_954_089_357]
    assert D <= len(primes)
    return torch.tensor(primes[:D], dtype=torch.long)


def _fast_hash(coords: torch.Tensor, primes: torch.Tensor, hash_size: int) -> torch.Tensor:
    """XOR-mix spatial hash to [0,hash_size). coords are integer grid corners."""
    x = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
    for d in range(coords.shape[-1]):
        x ^= coords[..., d] * primes[d]
    x ^= (x >> 32)
    if hash_size & (hash_size - 1) == 0:  # power-of-two -> bitmask
        return x & (hash_size - 1)
    return x.remainder(hash_size)


# -------------------------------
# Hash-grid encoders (D = 1 or 3)
# -------------------------------

class MultiResHashEncoderND(nn.Module):
    """Multi-resolution hash-grid encoder for D-dimensional inputs (Instant-NGP style).

    Input x must be normalized to [0,1]^D.
    """
    def __init__(
        self,
        D: int,
        num_levels: int = 16,
        level_dim: int = 2,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
        hash_size: int = 2 ** 19,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        padding: str = "wrap",  # "wrap" or "clamp"
    ):
        super().__init__()
        assert D in (1, 3), "Use D=1 for time or D=3 for (x,y,z)."
        self.D = D
        self.L = num_levels
        self.F = level_dim
        self.base = base_resolution
        self.growth = per_level_scale
        self.hash_size = hash_size
        self.padding = padding
        self.register_buffer("primes", _primes_for_hash(self.D))
        self.register_buffer("offsets", _make_offsets(self.D))  # (2^D, D)

        tables = []
        for _ in range(self.L):
            table = nn.Parameter(1e-4 * torch.randn(hash_size, self.F, dtype=dtype, device=device))
            tables.append(table)
        self.tables = nn.ParameterList(tables)

        res = [int(math.floor(self.base * (self.growth ** l))) for l in range(self.L)]
        self.register_buffer("resolutions", torch.tensor(res, dtype=torch.long))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        assert x01.shape[-1] == self.D
        x01 = x01.clamp(0.0, 1.0)
        N = x01.shape[0]
        feats = []
        for l in range(self.L):
            Rl = int(self.resolutions[l].item())
            table = self.tables[l]

            xf = x01 * Rl
            x0 = torch.floor(xf).to(torch.long)
            w = (xf - x0.to(xf.dtype))

            corners = (x0[:, None, :] + self.offsets[None, :, :])
            if self.padding == "wrap":
                corners = corners.remainder(Rl)
            else:
                corners = corners.clamp(0, Rl - 1)

            idx = _fast_hash(corners.reshape(-1, self.D), self.primes, self.hash_size).reshape(N, -1)
            corner_feats = table[idx]  # (N, 2^D, F)

            weights = 1.0
            for d in range(self.D):
                wd = w[:, d:d+1]  # (N,1)
                bd = self.offsets[:, d].to(x01.device).float()[None, :]  # (1,2^D)
                wd_full = torch.where(bd > 0.5, wd, 1.0 - wd)  # (N,2^D)
                weights = weights * wd_full

            feats.append((corner_feats * weights[..., None]).sum(dim=1))  # (N,F)

        return torch.cat(feats, dim=-1)  # (N, L*F)


# -------------------------------
# Time encoders
# -------------------------------

class FourierTimeEncoder(nn.Module):
    """Sin/Cos time features on [0,1], geometric freq progression, with optional
    integrated (anti-aliased) encoding over finite tick width Δt.

    If integrated=True, each feature is scaled by sinc(π f Δ) where Δ is the
    normalized tick width (Δ = tick_width / window_span).
    """
    def __init__(self, n_frequencies: int = 48, include_input: bool = False,
                 min_freq: float = 1.0, max_freq: float = 500.0,
                 integrated: bool = True, tick_width01: float = 1.0/1000.0):
        super().__init__()
        self.include_input = include_input
        self.integrated = integrated
        self.tick_width01 = tick_width01
        freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), n_frequencies)
        self.register_buffer("freqs", freqs)
        self.out_dim = (2 * n_frequencies) + (1 if include_input else 0)

    def forward(self, t01: torch.Tensor) -> torch.Tensor:
        # t01: (N,1) or (T,1) in [0,1]
        if t01.ndim == 1:
            t01 = t01[:, None]
        ang = 2 * math.pi * t01 @ self.freqs.view(1, -1)  # (N, F)
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.integrated:
            # Apply box-kernel integration attenuation per frequency (sinc)
            # sinc(x) = sin(x)/x, handle x≈0 with 1
            delta = self.tick_width01
            x = math.pi * delta * self.freqs  # (F,)
            # broadcast to both sin/cos halves
            s = torch.where(x.abs() < 1e-6, torch.ones_like(x), torch.sin(x) / x)
            s = torch.cat([s, s], dim=0)  # (2F,)
            feat = feat * s
        if self.include_input:
            feat = torch.cat([t01, feat], dim=-1)
        return feat


# Optional: 1D hash time encoder (kept for completeness; Fourier is faster/leaner)
class HashTimeEncoder(nn.Module):
    def __init__(self, num_levels=12, level_dim=2, base_resolution=64, per_level_scale=1.5, hash_size=2**18):
        super().__init__()
        self.encoder = MultiResHashEncoderND(D=1, num_levels=num_levels, level_dim=level_dim,
                                             base_resolution=base_resolution, per_level_scale=per_level_scale,
                                             hash_size=hash_size)
        self.out_dim = num_levels * level_dim

    def forward(self, t01: torch.Tensor) -> torch.Tensor:
        if t01.ndim == 1:
            t01 = t01[:, None]
        return self.encoder(t01)

# -------------------------------
# Factorized PMT Neural Field
# -------------------------------

@dataclass
class FieldConfig:
    # 3D hash grid (xyz)
    num_levels_xyz: int = 16
    level_dim_xyz: int = 2
    base_resolution_xyz: int = 16
    per_level_scale_xyz: float = 1.5
    hash_size_xyz: int = 2 ** 19
    padding_xyz: str = "clamp"  # bounded detector volumes → clamp
    # Time encoder (configured for 100 ns window, 0.1 ns ticks)
    time_encoder: str = "fourier"  # "fourier" or "hash1d"
    time_n_frequencies: int = 48    # 48→64 is typical
    time_min_freq: float = 1.0      # cycles per window
    time_max_freq: float = 500.0    # Nyquist for 1000 ticks across window
    time_integrated: bool = True
    # or for hash1d (rarely needed now)
    num_levels_t: int = 12
    level_dim_t: int = 2
    base_resolution_t: int = 64
    per_level_scale_t: float = 1.5
    hash_size_t: int = 2 ** 18
    # PMT embeddings / static features
    pmt_emb_dim: int = 8
    pmt_static_dim: int = 0
    # Geometry→latent MLP
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "silu"
    # Output head
    output_mode: str = "amplitude"  # or "log_amplitude"
    softplus_beta: float = 1.0
    # Causality options (optional)
    use_causal_gate: bool = False
    refractive_index: float = 1.38
    t_min: float = 0.0             # seconds for t01=0
    t_max: float = 100e-9          # seconds for t01=1 (100 ns)
    tick_width: float = 0.1e-9     # 0.1 ns
  # physical seconds corresponding to t01=1


class PMTNeuralFieldFactorized(nn.Module):
    """Factorized Instant-NGP field: encode (x,y,z) once, project with time features via bilinear form.

    Inputs are normalized to [0,1]. Provide scene_box/pmt_positions only if using causal gate.
    """
    def __init__(self, n_pmts: int, cfg: FieldConfig,
                 pmt_static: Optional[torch.Tensor] = None,
                 scene_box: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 pmt_positions: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.n_pmts = n_pmts

        # 3D encoder for (x,y,z)
        self.enc_xyz = MultiResHashEncoderND(
            D=3,
            num_levels=cfg.num_levels_xyz,
            level_dim=cfg.level_dim_xyz,
            base_resolution=cfg.base_resolution_xyz,
            per_level_scale=cfg.per_level_scale_xyz,
            hash_size=cfg.hash_size_xyz,
            padding=cfg.padding_xyz,
        )
        self.xyz_feat_dim = cfg.num_levels_xyz * cfg.level_dim_xyz

        # Time encoder
        if cfg.time_encoder == "fourier":
            tick_w01 = (cfg.tick_width / max(cfg.t_max - cfg.t_min, 1e-12))
            self.enc_t = FourierTimeEncoder(
                n_frequencies=cfg.time_n_frequencies,
                include_input=False,
                min_freq=cfg.time_min_freq,
                max_freq=cfg.time_max_freq,
                integrated=cfg.time_integrated,
                tick_width01=tick_w01,
            )
            self.t_feat_dim = self.enc_t.out_dim
        elif cfg.time_encoder == "hash1d":
            he = HashTimeEncoder(cfg.num_levels_t, cfg.level_dim_t, cfg.base_resolution_t,
                                 cfg.per_level_scale_t, cfg.hash_size_t)
            self.enc_t = he
            self.t_feat_dim = he.out_dim
            he = HashTimeEncoder(cfg.num_levels_t, cfg.level_dim_t, cfg.base_resolution_t,
                                 cfg.per_level_scale_t, cfg.hash_size_t)
            self.enc_t = he
            self.t_feat_dim = he.out_dim
        else:
            raise ValueError("time_encoder must be 'fourier' or 'hash1d'")

        # Per-PMT embedding and optional static features
        self.pmt_emb = nn.Embedding(n_pmts, cfg.pmt_emb_dim)
        self.register_buffer("pmt_static", pmt_static if pmt_static is not None else None)

        # Geometry/PMT head -> latent H
        in_dim = self.xyz_feat_dim + cfg.pmt_emb_dim + (cfg.pmt_static_dim or 0) + (1 if self.pmt_positions is not None else 0)
        layers = []
        last = in_dim
        act = nn.SiLU() if cfg.activation == "silu" else nn.ReLU()
        for _ in range(cfg.num_layers):
            layers += [nn.Linear(last, cfg.hidden_dim), act]
            last = cfg.hidden_dim
        self.geom_mlp = nn.Sequential(*layers)

        # Bilinear fusion: H (N, Dh) × M (Dh×Dt) × Tfeat^T (Dt×T) -> (N,T)
        self.bilinear = nn.Parameter(0.01 * torch.randn(cfg.hidden_dim, self.t_feat_dim))
        # Optional additive biases (helpful in practice)
        self.bias = nn.Parameter(torch.zeros(1))
        self.h_bias = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.t_bias = nn.Parameter(torch.zeros(self.t_feat_dim))

        # Output nonlinearity config
        self.softplus = nn.Softplus(beta=cfg.softplus_beta)

        # For causal gating
        self.register_buffer("scene_min", None)
        self.register_buffer("scene_max", None)
        self.register_buffer("pmt_positions", None)
        if cfg.use_causal_gate:
            assert scene_box is not None and pmt_positions is not None, (
                "Causal gate requires scene_box=(min,max) and pmt_positions.")
            self.scene_min = scene_box[0].float()
            self.scene_max = scene_box[1].float()
            self.pmt_positions = pmt_positions.float()

    # ---- helpers ----
    @staticmethod
    def _unnormalize_coords(coords01: torch.Tensor, scene_min: torch.Tensor, scene_max: torch.Tensor) -> torch.Tensor:
        return scene_min + coords01 * (scene_max - scene_min)

    def _tof_seconds(self, xyz01: torch.Tensor, pmt_id: torch.Tensor) -> torch.Tensor:
        # xyz01: (N,3) in [0,1]
        xyz = self._unnormalize_coords(xyz01, self.scene_min, self.scene_max)  # (N,3)
        p = self.pmt_positions[pmt_id]
        d = torch.norm(p - xyz, dim=-1)  # meters
        c = 299.792_458 # mm/ns
        v = c / self.cfg.refractive_index
        return d / v  # seconds

    def _gate(self, xyz01: torch.Tensor, pmt_id: torch.Tensor, t01: torch.Tensor) -> torch.Tensor:
        # t01: (N,1) or (T,) -> broadcast to (N,T)
        t_phys = self.cfg.t_min + t01 * (self.cfg.t_max - self.cfg.t_min)
        if t_phys.dim() == 1:
            t_phys = t_phys[None, :]  # (1,T)
        t0 = self._tof_seconds(xyz01, pmt_id).unsqueeze(-1)  # (N,1)
        # 1 ns default works; if your t is in seconds, set t_max - t_min accordingly
        sigma = 1e-9
        return torch.sigmoid((t_phys - t0) / sigma)

    # ---- encoders ----
    def encode_xyz_pmt(self, xyz01: torch.Tensor, pmt_id: torch.Tensor) -> torch.Tensor:
        exyz = self.enc_xyz(xyz01)

        feats = [exyz, self.pmt_emb(pmt_id)] 
        if self.pmt_positions is not None: # use tof as a feature
            t0 = self._tof_seconds(xyz01, pmt_id)
            t0_norm = (t0 - self.cfg.t_min) / (self.cfg.t_max - self.cfg.t_min)
            feats.append(t0_norm)
        if self.cfg.pmt_static_dim and self.pmt_static is not None: # use static features
            feats.append(self.pmt_static[pmt_id])
        h = self.geom_mlp(torch.cat(feats, dim=-1))  # (N, Dh)
        return h

    def encode_time(self, t01: torch.Tensor) -> torch.Tensor:
        if t01.ndim == 1:
            t01 = t01[:, None]
        return self.enc_t(t01)  # (T, Dt)


    # ---- forward APIs ----
    def forward_waveform(self, xyz01: torch.Tensor, pmt_id: torch.Tensor, t01_vec: torch.Tensor) -> torch.Tensor:
        """Return A for all times: (N, T). xyz01: (N,3), pmt_id: (N,), t01_vec: (T,) or (T,1)."""
        H = self.encode_xyz_pmt(xyz01, pmt_id)  # (N, Dh)
        Tfeat = self.encode_time(t01_vec)       # (T, Dt)
        # add small per-branch biases
        Hb = H + self.h_bias
        Tb = Tfeat + self.t_bias
        raw = Hb @ self.bilinear @ Tb.T  # (N, T)
        raw = raw + self.bias
        if self.cfg.output_mode == "log_amplitude":
            out = raw
        else:
            out = self.softplus(raw)
        if self.cfg.use_causal_gate:
            gate = self._gate(xyz01, pmt_id, t01_vec)  # (N,T)
            out = out * gate
        return out  # (N,T)

    def forward_sample(self, coords01: torch.Tensor, pmt_id: torch.Tensor) -> torch.Tensor:
        """Per-sample forward for training with scattered (x,y,z,t). coords01: (N,4)."""
        xyz01 = coords01[:, :3]
        t01 = coords01[:, 3:4]
        H = self.encode_xyz_pmt(xyz01, pmt_id)           # (N, Dh)
        Tfeat = self.encode_time(t01).squeeze(1)         # (N, Dt)
        raw = (H + self.h_bias) @ self.bilinear @ (Tfeat + self.t_bias).T  # (N,N)
        raw = raw.diag().unsqueeze(-1)  # keep only matching pairs
        raw = raw + self.bias
        if self.cfg.output_mode == "log_amplitude":
            out = raw
        else:
            out = self.softplus(raw)
        if self.cfg.use_causal_gate:
            out = out * self._gate(xyz01, pmt_id, t01).unsqueeze(-1)
        return out  # (N,1)


# -------------------------------
# Losses & minimal training loop
# -------------------------------

def poisson_deviance(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)
    return (pred - target + target * (target.add(eps).log() - pred.log())).mean()


def log_mse(pred_log: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (pred_log - (target.add(eps).log())).pow(2).mean()


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 20_000
    batch_size: int = 131_072
    loss_type: str = "poisson"  # or "logmse"
    amp: bool = True


class LUTDataset(torch.utils.data.Dataset):
    """Wrap precomputed LUT samples for scattered training.

    coords01: (N,4) in [0,1]
    pmt_id:   (N,)
    amplitude:(N,)
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


def train_factorized(model: PMTNeuralFieldFactorized, ds: LUTDataset, tcfg: TrainConfig, device: str = "cuda"):
    model = model.to(device)
    loader = torch.utils.data.DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True,
                                         pin_memory=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=(tcfg.amp and device.startswith("cuda")))
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
                out = model.forward_sample(coords01, pmt_id)
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
# HDF5-backed Dataset + TOF-aware Sampler
# -------------------------------
try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    h5py = None


class H5WaveformLUTDataset(torch.utils.data.Dataset):
    """Random-access view over an HDF5 LUT with shape (N_vox, N_pmts, N_ticks) or flattened.

    Required datasets in the H5 file:
      - 'vox_xyz': float32, shape (N_vox, 3) in physical units (meters)
      - one of:
          * 'amplitudes': float32, shape (N_vox, N_pmts, N_ticks)
          * 'amplitudes_flat': float32, shape (N_vox, N_pmts*N_ticks)
    Optional datasets:
      - 'pmt_pos': float32, shape (N_pmts, 3)
      - 'pmt_normal': float32, shape (N_pmts, 3)

    __getitem__ expects a triple index (vox_idx, pmt_idx, tick_idx).
    It returns: coords01(4,), pmt_id, amplitude(float)
    """

    def __init__(
        self,
        h5_path: str,
        scene_min: torch.Tensor,
        scene_max: torch.Tensor,
        t_min: float,
        t_max: float,
    ):
        assert h5py is not None, "h5py is required for H5WaveformLUTDataset"
        self.h5_path = h5_path
        self.scene_min = scene_min.clone().float()
        self.scene_max = scene_max.clone().float()
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        # Lazy-open per worker
        self._f = None
        with h5py.File(h5_path, "r") as f:
            self._vox_xyz = torch.from_numpy(
                f["vox_xyz"][...]
            ).float()  # keep in RAM (N_vox,3)
            if "amplitudes" in f:
                shp = f["amplitudes"].shape
                self.N_vox, self.N_pmts, self.N_ticks = shp
                self.flat = False
            else:
                shp = f["amplitudes_flat"].shape
                self.N_vox, flat_dim = shp
                # infer pmts and ticks from attrs
                self.N_pmts = int(f["amplitudes_flat"].attrs.get("N_pmts", 81))
                self.N_ticks = int(
                    f["amplitudes_flat"].attrs.get("N_ticks", flat_dim // self.N_pmts)
                )
                self.flat = True
            # Optional PMT geometry (not used by dataset directly; samplers may query via accessors)
            self._pmt_pos = (
                torch.from_numpy(f["pmt_pos"][...]).float() if "pmt_pos" in f else None
            )
            self._pmt_normal = (
                torch.from_numpy(f["pmt_normal"][...]).float()
                if "pmt_normal" in f
                else None
            )

    # Helpers to open file lazily in each worker
    def _ensure_file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
        return self._f

    @property
    def vox_xyz(self) -> torch.Tensor:
        return self._vox_xyz

    @property
    def pmt_pos(self) -> Optional[torch.Tensor]:
        return self._pmt_pos

    @property
    def pmt_normal(self) -> Optional[torch.Tensor]:
        return self._pmt_normal

    def __len__(self):
        # Length is not used when paired with a custom Sampler; return N_vox for formality
        return self.N_vox

    def __getitem__(self, index):
        # Expect index as (vox_idx, pmt_idx, tick_idx)
        vox_idx, pmt_idx, tick_idx = index
        f = self._ensure_file()
        if self.flat:
            off = pmt_idx * self.N_ticks + tick_idx
            amp = f["amplitudes_flat"][vox_idx, off]
        else:
            amp = f["amplitudes"][vox_idx, pmt_idx, tick_idx]
        amp = float(amp)
        # Normalize xyz and t
        xyz = self._vox_xyz[vox_idx]
        xyz01 = (xyz - self.scene_min) / (self.scene_max - self.scene_min)
        t01 = tick_idx / (self.N_ticks - 1)
        coords01 = torch.cat([xyz01, torch.tensor([t01], dtype=torch.float32)])
        return (
            coords01,
            torch.tensor(pmt_idx, dtype=torch.long),
            torch.tensor(amp, dtype=torch.float32),
        )


class RandomTOFSampler(torch.utils.data.Sampler):
    """TOF-aware sampler that yields (vox_idx, pmt_idx, tick_idx) triples.

    It mixes three strategies per batch:
      - near-peak around the analytic TOF bin (if PMT positions are provided)
      - log-tail: Δt drawn log-uniform up to window
      - uniform over ticks

    Parameters
    ----------
    dataset: H5WaveformLUTDataset
    steps_per_epoch: number of *batches* you plan per epoch (for epoch sizing)
    batch_size: yielded triples per batch
    mix: dict with fractions, e.g., {"peak":0.5, "tail":0.3, "uniform":0.2}
    peak_ns: half-width (ns) around the TOF bin for peak sampling
    tail_min_ns, tail_max_ns: Δt range for log-tail sampling (ns)
    refractive_index: n for TOF in LAr
    """

    def __init__(
        self,
        dataset: H5WaveformLUTDataset,
        steps_per_epoch: int,
        batch_size: int,
        mix={"peak": 0.5, "tail": 0.3, "uniform": 0.2},
        peak_ns: float = 0.5,
        tail_min_ns: float = 0.5,
        tail_max_ns: float = 100.0,
        refractive_index: float = 1.38,
        seed: int = 12345,
    ):
        self.ds = dataset
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.mix = mix
        self.peak_ns = peak_ns
        self.tail_min_ns = tail_min_ns
        self.tail_max_ns = tail_max_ns
        self.n = refractive_index
        self.rng = torch.Generator().manual_seed(seed)
        # precompute helpers
        self.dt = (self.ds.t_max - self.ds.t_min) / (
            self.ds.N_ticks - 1
        )  # seconds per tick
        self.ns_per_tick = self.dt * 1e9
        self.peak_half_bins = max(1, int(round(self.peak_ns / self.ns_per_tick)))
        self.tail_min_bins = max(1, int(round(self.tail_min_ns / self.ns_per_tick)))
        self.tail_max_bins = max(1, int(round(self.tail_max_ns / self.ns_per_tick)))
        self.c = 299_792_458.0
        # Cache in-RAM tensors for speed
        self.vox_xyz = self.ds.vox_xyz  # (N_vox,3)
        self.pmt_pos = self.ds.pmt_pos  # (N_pmts,3) or None

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def _sample_vox_pmt(self, k: int):
        Nvox, Npmt = self.ds.N_vox, self.ds.N_pmts
        # randint on CPU generator
        vox = int(torch.randint(0, Nvox, (1,), generator=self.rng).item())
        pmt = int(torch.randint(0, Npmt, (1,), generator=self.rng).item())
        return vox, pmt

    def _tof_bin(self, vox_idx: int, pmt_idx: int) -> int:
        if self.pmt_pos is None:
            # Fallback: center of window
            return (self.ds.N_ticks - 1) // 2
        x = self.vox_xyz[vox_idx]
        p = self.pmt_pos[pmt_idx]
        d = torch.norm(p - x).item()
        v = self.c / self.n
        t0 = d / v  # seconds
        # map to bin
        b = int(
            round(
                (t0 - self.ds.t_min)
                / (self.ds.t_max - self.ds.t_min)
                * (self.ds.N_ticks - 1)
            )
        )
        return max(0, min(self.ds.N_ticks - 1, b))

    def _sample_tick_peak(self, b0: int) -> int:
        low = max(0, b0 - self.peak_half_bins)
        high = min(self.ds.N_ticks - 1, b0 + self.peak_half_bins)
        return int(torch.randint(low, high + 1, (1,), generator=self.rng).item())

    def _sample_tick_tail(self, b0: int) -> int:
        # log-uniform Δ in [tail_min, tail_max] bins, then add to b0
        lo = self.tail_min_bins
        hi = self.tail_max_bins
        if hi <= lo:
            hi = lo + 1
        # sample u~U(0,1), Δ = round(exp(log(lo)+u*(log(hi)-log(lo))))
        u = torch.rand(1, generator=self.rng).item()
        Δ = int(round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))))
        b = b0 + Δ
        if b >= self.ds.N_ticks:
            b = self.ds.N_ticks - 1
        return b

    def _sample_tick_uniform(self) -> int:
        return int(torch.randint(0, self.ds.N_ticks, (1,), generator=self.rng).item())

    def __iter__(self):
        # Worker sharding
        info = torch.utils.data.get_worker_info()
        if info is not None:
            # derive per-worker RNG from base + worker id
            base_seed = self.rng.initial_seed()
            self.rng = torch.Generator().manual_seed(base_seed + info.id + 1)

        num = self.__len__()
        for i in range(num):
            vox, pmt = self._sample_vox_pmt(i)
            # choose mode
            u = torch.rand(1, generator=self.rng).item()
            m_peak = self.mix.get("peak", 0.0)
            m_tail = self.mix.get("tail", 0.0)
            if u < m_peak:
                b0 = self._tof_bin(vox, pmt)
                tick = self._sample_tick_peak(b0)
            elif u < m_peak + m_tail:
                b0 = self._tof_bin(vox, pmt)
                tick = self._sample_tick_tail(b0)
            else:
                tick = self._sample_tick_uniform()
            yield (vox, pmt, tick)

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detector volume: center (-1,0,0), extents (2,4,4) → scene_min=(-2,-2,-2), scene_max=(0,2,2)
    scene_min = torch.tensor([-2.0, -2.0, -2.0])
    scene_max = torch.tensor([ 0.0,  2.0,  2.0])

    def normalize_xyz(xyz):
        return (xyz - scene_min) / (scene_max - scene_min)

    # Time window: 0 → 100 ns with 0.1 ns ticks
    t_min, t_max, n_ticks = 0.0, 100e-9, 1000
    t01_vec = torch.linspace(0.0, 1.0, steps=n_ticks, device=device)

    n_pmts = 81

    # Fake PMT positions/normals (replace with real; only needed for causal gate)
    pmt_pos = torch.randn(n_pmts, 3)

    cfg = FieldConfig(
        num_levels_xyz=16, level_dim_xyz=2, base_resolution_xyz=16, per_level_scale_xyz=1.5, hash_size_xyz=2**19,
        padding_xyz="clamp",
        time_encoder="fourier", time_n_frequencies=48, time_min_freq=1.0, time_max_freq=500.0,
        time_integrated=True,
        pmt_emb_dim=8, pmt_static_dim=0,
        hidden_dim=64, num_layers=2, activation="silu",
        output_mode="amplitude", softplus_beta=1.0,
        use_causal_gate=False, refractive_index=1.38,
        t_min=t_min, t_max=t_max, tick_width=0.1e-9,
    )

    model = PMTNeuralFieldFactorized(n_pmts=n_pmts, cfg=cfg)

    # Synthetic scattered samples for training demo
    N = 100_000
    xyz_phys = scene_min + (scene_max - scene_min) * torch.rand(N, 3)
    coords01 = torch.cat([normalize_xyz(xyz_phys), torch.rand(N, 1)], dim=1)
    pmt_id = torch.randint(0, n_pmts, (N,))

    # Fake target amplitudes
    t01 = coords01[:, 3:4]
    target = (
        4.0 * torch.exp(-((t01 - 0.20) ** 2) / (2 * (0.002) ** 2)) +
        2.5 * torch.exp(-((t01 - 0.55) ** 2) / (2 * (0.003) ** 2)) + 0.02
    ).squeeze(-1)

    ds = LUTDataset(coords01, pmt_id, target)
    tcfg = TrainConfig(lr=1e-3, steps=1500, batch_size=32768, loss_type="poisson", amp=True)
    train_factorized(model, ds, tcfg, device=device)

    # Vectorized waveform for a few voxels/PMTs
    with torch.no_grad():
        xyz_batch = normalize_xyz(scene_min + (scene_max - scene_min) * torch.rand(4, 3)).to(device)
        pmts = torch.tensor([0, 7, 13, 42], device=device)
        A = model.forward_waveform(xyz_batch, pmts, t01_vec)
        print("waveform shape:", A.shape)  # (4, 1000)
