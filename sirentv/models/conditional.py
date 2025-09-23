import numpy as np
import torch
from slar.base import Siren
from slar.transform import partial_xform_vis
from torch import nn
from sirentv.models.builder import MODELS

@MODELS.register_module()
class ConditionalSiren(nn.Module):
    def __init__(
        self,
        xform_vis: dict,
        in_features=3,
        hidden_features_pos=512,
        hidden_layers_pos=2,
        hidden_features_time=128,
        hidden_layers_time=1,
        embedding_dim=128,
        n_pmts=81,
        n_time_steps=1000,
        final_activation: str = "sigmoid",  # or "sine"
        # low rank fusion options
        use_low_rank_fusion: bool = False,
        rank_vis=8,
        rank_time=8,
        fusion_omega0: float = 30.0,
        pmt_pos_path: str = None,
        steepness_factor: float = 10.0,
        use_CDF: True
    ):
        super().__init__()
        self.use_low_rank_fusion = use_low_rank_fusion
        self.final_activation = final_activation
        self.fusion_omega0 = float(fusion_omega0)
        self.pos_enc = Siren(
            in_features=in_features,
            hidden_features=hidden_features_pos,
            hidden_layers=hidden_layers_pos,
            out_features=embedding_dim,
            outermost_linear=False,
        )
        self.tenc = Siren(
            in_features=1,
            hidden_features=hidden_features_time,
            hidden_layers=hidden_layers_time,
            out_features=embedding_dim,
            outermost_linear=False,
        )
        self.pmt_enc = nn.Embedding(n_pmts, embedding_dim)

        # use a shared projection without bias; add a single scalar bias at the end
        self.v_proj = nn.Linear(embedding_dim, 1, bias=False)
        self.t_proj = nn.Linear(embedding_dim, 1, bias=False)
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        self.n_pmts = int(n_pmts)
        
        if self.use_low_rank_fusion:
            self.rank_vis = int(rank_vis)
            self.rank_time = int(rank_time)
            # self.pos_to_rank_vis = nn.Linear(embedding_dim, self.rank_vis, bias=False)
            # self.pmt_to_rank_vis = nn.Linear(embedding_dim, self.rank_vis, bias=False)
            # self.bias_v = nn.Parameter(torch.tensor(0.0))
            self.pos_to_rank_time = nn.Linear(embedding_dim, self.rank_time, bias=False)
            self.pmt_to_rank_time = nn.Linear(embedding_dim, self.rank_time, bias=False)
            self.time_to_rank_time = nn.Linear(embedding_dim, self.rank_time, bias=False)
            self.bias_t = nn.Parameter(torch.tensor(0.0))

            self.G = nn.Parameter(torch.empty(self.rank_time, self.rank_time, self.rank_time))
            torch.nn.init.normal_(self.G, mean=0, std=1.0 / self.rank_time ** 1.5)

            # initialize for sine activation if requested
            if self.final_activation == "sine":
                with torch.no_grad():
                    w_base = (6.0 / float(embedding_dim)) ** 0.5
                    # for layer in [self.pos_to_rank_vis, self.pmt_to_rank_vis]:
                    #     w = w_base / self.fusion_omega0 ** (1.0/2.0)
                    #     layer.weight.uniform_(-w, w)
                    for layer in [self.pos_to_rank_time, self.pmt_to_rank_time, self.time_to_rank_time]:
                        w = w_base / self.fusion_omega0 ** (1.0/3.0)
                        layer.weight.uniform_(-w, w)

        ticks = torch.linspace(-1, 1, n_time_steps)
        self.register_buffer("ticks", ticks)
        self.register_buffer("pmt_ids", torch.arange(n_pmts, dtype=torch.long))

        # pmt positions in same coord space as x
        pmt_pos = torch.zeros(self.n_pmts, 3) if pmt_pos_path is None else torch.from_numpy(np.loadtxt(pmt_pos_path, delimiter=',')).float()
        self.register_buffer("pmt_pos", pmt_pos, persistent=False)

        if pmt_pos.eq(0).all():
            self.pmt_geom_proj = None
        else:
            self.pmt_geom_proj = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self._xform_vis, self._inv_xform_vis = partial_xform_vis(xform_vis)

        #self.out_features = [n_pmts, n_pmts * n_time_steps]
        self.out_features = [1, 1 + n_time_steps]
        self._steepness_factor = steepness_factor
        self._use_CDF = use_CDF

    def set_pmt_positions(self, pmt_pos: torch.Tensor):
        assert pmt_pos.shape == (self.n_pmts, 3)
        self.pmt_pos = pmt_pos.to(self.ticks.device).float()

    def forward(self, x, t=None):
        if t is None:
            t = self.ticks.unsqueeze(1)
        assert t.ndim == 2 and t.shape[1] == 1, f"t.shape: {t.shape} != (n, 1)"
        assert x.ndim == 2 and x.shape[1] == 3, f"x.shape: {x.shape} != (n, 3)"
        assert torch.all((-1 <= x) & (x <= 1)), "x is not in [-1, 1]"

        # encode position, time, and pmt separately
        x_feat = self.pos_enc(x)
        t_feat = self.tenc(t)

        N = x_feat.shape[0]
        T = t.shape[0]
        PMT = self.pmt_ids.shape[0]
        pmt_feat = self.pmt_enc(self.pmt_ids).unsqueeze(0)
        if self.pmt_geom_proj is not None:
            d = self.pmt_pos.unsqueeze(0) - x.unsqueeze(1)
            r = torch.linalg.norm(d, dim=-1).clamp_min(1e-6)  # coord units
            inv_r = 1.0 / r
            inv_r2 = inv_r * inv_r
            costheta = (d * torch.tensor([1,0,0], device=d.device) / r).sum(-1)
            geom = torch.stack([r, inv_r, inv_r2, costheta], dim=-1)
            # compute angle of incidence using a normal of [1,0,0]
            pmt_feat = pmt_feat + self.pmt_geom_proj(geom)

        if self.use_low_rank_fusion:
            # fuse via contraction over embed dim, out_t = ux_t @ up_t @ ut_t / rank_time ** (1/3)
            ux_t = self.pos_to_rank_time(x_feat)
            up_t = self.pmt_to_rank_time(pmt_feat).squeeze()
            ut_t = self.time_to_rank_time(t_feat)
            out_t = torch.einsum('ir,js,kt,rst->ijk', ux_t, up_t, ut_t, self.G)
            out_t = out_t + self.bias_t
        else:
            # simple linear fusion, out_t = linear(x_t + p_t + t_t)
            x_t = self.t_proj(x_feat).squeeze(-1)
            t_t = self.t_proj(t_feat).squeeze(-1)
            p_t = self.t_proj(pmt_feat).squeeze(-1)
            out_t = (
                x_t.view(-1, 1, 1)
                + p_t.view(1, -1, 1)
                + t_t.view(1, 1, -1)
                + self.output_bias
            )
        if self.final_activation == "sine":
            out_t = torch.sin(self.fusion_omega0 * out_t).view(N*PMT, T)
        else:
            out_t = out_t.sigmoid().view(N*PMT,T)

        out_v = self._xform_vis(
            self._inv_xform_vis(out_t.reshape(N, PMT, T)).sum(-1)).unsqueeze(1)

        output = dict(v=out_v, t=out_t)
        return output