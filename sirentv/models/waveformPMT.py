import math
from typing import List, Literal, Union

import numpy as np
import torch
from photonlib import AABox
from slar.base import Siren
from slar.transform import partial_xform_vis
from torch import nn

from sirentv.models import MODELS


@MODELS.register_module()
class WaveformSiren(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        out_features: int = 1000,
        outermost_linear: bool = False,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        xform_vis: dict = {}, # no op, but always required
    ):
        super().__init__()

        hidden_features = (
            [hidden_features] if isinstance(hidden_features, int) else hidden_features
        )
        hidden_layers = (
            [hidden_layers] if isinstance(hidden_layers, int) else hidden_layers
        )
        out_features = [out_features] if isinstance(out_features, int) else out_features

        print("=" * 20, "Encoder", "=" * 20)
        self.encoder = Siren(
            in_features=in_features,
            hidden_features=hidden_features[0],
            hidden_layers=hidden_layers[0] - 1,
            out_features=hidden_features[0],
            outermost_linear=False,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
        print("=" * 20, "Waveform decoder", "=" * 20)
        self.waveform_decoder = Siren(
            in_features=hidden_features[0],
            hidden_features=hidden_features[0],
            hidden_layers=hidden_layers[0] - 1,
            out_features=out_features[0],
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        self.hidden_omega_0 = hidden_omega_0

        self.init_weights()
        self.out_features = self.waveform_decoder.net[-1].out_features

    def init_weights(self):
        """
        Siren initializes all first layer weights with a uniform distribution, and not the
        custom distribution often used for SIREN layers.
        """
        with torch.no_grad():
            for layer in self.encoder.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.uniform_(
                        -np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                        np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                    )
                else:
                    layer.init_weights()

            for decoder in [self.waveform_decoder]:
                for layer in decoder.net:
                    if isinstance(layer, nn.Linear):
                        layer.weight.uniform_(
                            -np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                            np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                        )
                    else:
                        layer.is_first = False
                        layer.init_weights()

            assert all(not layer.is_first for layer in self.encoder.net[1:])

            assert all(
                not layer.is_first
                for layer in self.waveform_decoder.net
                if not isinstance(layer, nn.Linear)
            )

    def forward(self, coords, clone=False):
        if clone:
            coords = coords.clone().detach().requires_grad_(True)

        x = self.encoder(coords)
        waveform = self.waveform_decoder(x)
        visibility = torch.sum(waveform, dim=-1)
        return torch.cat([visibility, waveform], dim=-1)

    def unfreeze_all(self):
        """Unfreeze all parameters in the network"""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Return only the parameters that require gradients"""
        return filter(lambda p: p.requires_grad, self.parameters())

    def freeze_all_but_(self, decoder: Literal["timing", "visibility"] = "timing"):
        self.unfreeze_all()
        self.position_encoder.requires_grad_(False)
        if decoder == "timing":
            self.visibility_decoder.requires_grad_(False)
        elif decoder == "visibility":
            self.waveform_decoder.requires_grad_(False)
        else:
            raise ValueError(f"Invalid decoder: {decoder}")

    def print_trainable_params(self):
        """Print the names of trainable parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")

    def __repr__(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        memory_mb = n_params * 4 / (1024 * 1024)  # assume fp32
        return f"{n_params:,} trainable parameters\n{memory_mb:2f} MB\n{super().__repr__()}"