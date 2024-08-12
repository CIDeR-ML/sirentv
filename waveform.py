import torch
import torch.nn as nn
from larndsim.const import light
import math
import numpy as np
import os
import yaml
import torch.nn.functional as F
from typing import TypeVar, Dict

T = TypeVar("T")

def print_grad(name):
    def hook(grad):
        print(f"Gradient for {name}: {grad}")

    return hook


class Config(Dict[str, T]):
    def __getattr__(self, name: str) -> T:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: T) -> None:
        self[name] = value

def fraction_to_softmax_params(fraction):
    # Calculate the required input to the softmax function
    x = np.log(fraction / (1 - fraction))

    # Create the input tensor
    inputs = torch.tensor([x, 0.0], dtype=torch.float32)
    return inputs


class BatchedLightSimulation(nn.Module):
    def __init__(self, ndet=48, cfg=light, verbose=False):
        super().__init__()
        self.ndet = ndet

        if isinstance(cfg, str):
            cfg = yaml.safe_load(open(cfg,'r').read())
            cfg = Config(**cfg)

        # Differentiable parameters
        self.singlet_fraction_logit = nn.Parameter(torch.tensor(np.log(cfg.SINGLET_FRACTION / (1 - cfg.SINGLET_FRACTION)), dtype=torch.float32))
        self.log_tau_s = nn.Parameter(torch.tensor(np.log10(cfg.TAU_S), dtype=torch.float32))  # in ns
        self.log_tau_t = nn.Parameter(torch.tensor(np.log10(cfg.TAU_T), dtype=torch.float32))  # in ns
        self.light_oscillation_period = nn.Parameter(
            torch.tensor(cfg.LIGHT_OSCILLATION_PERIOD, dtype=torch.float32)
        )  # in ns
        self.light_response_time = nn.Parameter(
            torch.tensor(cfg.LIGHT_RESPONSE_TIME, dtype=torch.float32)
        )  # in ns
        self.light_gain = (
            torch.tensor(cfg.LIGHT_GAIN[:ndet], dtype=torch.float32)
        )  # One gain per detector

        # Constants
        self.light_tick_size = light.LIGHT_TICK_SIZE
        self.light_window = [1, 100]

        self.conv_ticks = math.ceil(
            (self.light_window[1] - self.light_window[0]) / self.light_tick_size
        )

        # Create the scintillation model kernel
        self.time_ticks = torch.arange(self.conv_ticks)

        if verbose:
            self.register_grad_hook()

    def to(self, device):
        self.time_ticks = self.time_ticks.to(device)
        self.light_oscillation_period = self.light_oscillation_period.to(device)
        self.light_response_time = self.light_response_time.to(device)
        self.light_gain = self.light_gain.to(device)
        super().to(device)
        return self
    
    def register_grad_hook(self):
        for name, p in self.named_parameters():
            p.register_hook(print_grad(name))

    def scintillation_model(self, time_tick):
        """
        Calculates the fraction of scintillation photons emitted
        during time interval `time_tick` to `time_tick + 1`

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            light (object): object containing light-related constants

        Returns:
            torch.Tensor: fraction of scintillation photons
        """

        singlet_fraction = torch.sigmoid(self.singlet_fraction_logit)

        tau_s = torch.pow(10, self.log_tau_s)
        tau_t = torch.pow(10, self.log_tau_t)

        p1 = (
            singlet_fraction
            * torch.exp(-time_tick * self.light_tick_size / tau_s)
            * (1 - torch.exp(-self.light_tick_size / tau_s))
        )
        p3 = (
            (1 - singlet_fraction)
            * torch.exp(-time_tick * self.light_tick_size / tau_t)
            * (1 - torch.exp(-self.light_tick_size / tau_t))
        )
        return (p1 + p3) * (time_tick >= 0).float()

    def calc_scintillation_effect(self, light_sample_inc):
        """
        Applies a smearing effect due to the liquid argon scintillation time profile using
        a two decay component scintillation model.

        Args:
            light_sample_inc (torch.Tensor): shape `(ninput, ndet, ntick)`, light incident on each detector

        Returns:
            torch.Tensor: shape `(ninput, ndet, ntick)`, light incident on each detector after accounting for scintillation time
        """
        ninput, ndet, ntick = light_sample_inc.shape

        # Pad the input tensor
        padded_input = torch.nn.functional.pad(
            light_sample_inc, (self.conv_ticks - 1, 0)
        )

        # Reshape for grouped convolution
        padded_input = padded_input.view(1, ninput * ndet, -1)

        scintillation_kernel = self.scintillation_model(self.time_ticks).flip(0)

        # Create a separate kernel for each input and detector
        scintillation_kernel = scintillation_kernel.repeat(ninput * ndet, 1).view(
            ninput * ndet, 1, -1
        )

        # Perform the convolution
        light_sample_inc_scint = torch.nn.functional.conv1d(
            padded_input, scintillation_kernel, groups=ninput * ndet
        )

        # Trim the result to match the input shape
        light_sample_inc_scint = light_sample_inc_scint.view(ninput, ndet, -1)[
            :, :, :ntick
        ]

        return light_sample_inc_scint

    def sipm_response_model(self, time_tick):
        """
        Calculates the SiPM response from a PE at `time_tick` relative to the PE time

        Args:
            time_tick (torch.Tensor): time tick relative to t0

        Returns:
            torch.Tensor: response
        """
        t = time_tick * self.light_tick_size
        impulse = (
            (t >= 0).float()
            * torch.exp(-t / self.light_response_time)
            * torch.sin(t / self.light_oscillation_period)
        )
        # normalize to 1
        impulse /= self.light_oscillation_period * self.light_response_time**2
        impulse *= self.light_oscillation_period**2 + self.light_response_time**2
        return impulse * self.light_tick_size

    def calc_light_detector_response(self, light_sample_inc):
        """
        Simulates the SiPM response and digit

        Args:
            light_sample_inc (torch.Tensor): shape `(ninput, ndet, ntick)`, PE produced on each SiPM at each time tick

        Returns:
            torch.Tensor: shape `(ninput, ndet, ntick)`, ADC value at each time tick
        """
        ninput, ndet, ntick = light_sample_inc.shape

        # Pad the input tensor
        padded_input = torch.nn.functional.pad(
            light_sample_inc, (self.conv_ticks - 1, 0)
        )

        # Reshape for grouped convolution
        padded_input = padded_input.view(1, ninput * ndet, -1)

        light_response_kernel = self.sipm_response_model(self.time_ticks).flip(0)

        # Create a separate kernel for each input and detector
        light_response_kernel = light_response_kernel.repeat(ninput * ndet, 1).view(
            ninput * ndet, 1, -1
        )

        # Perform the convolution
        light_response = torch.nn.functional.conv1d(
            padded_input, light_response_kernel, groups=ninput * ndet
        )

        # Trim the result to match the input shape
        light_response = light_response.view(ninput, ndet, -1)[:, :, :ntick]

        # Apply the gain
        light_response = self.light_gain.view(1, -1, 1) * light_response

        return light_response

    def downsample_waveform(self, waveform, ns_per_tick=16):
        ninput, ndet, ntick = waveform.shape
        ntick_down = ntick // ns_per_tick
        downsample = waveform.view(
            ninput, ndet, ntick_down, ns_per_tick
        ).sum(dim=3)
        return downsample

    def forward(self, timing_dist):
        x = self.calc_scintillation_effect(timing_dist)
        x = self.calc_light_detector_response(x)
        x = self.downsample_waveform(x)

        return x
    

class TimingDistributionSampler:
    def __init__(self, cdf, output_shape):
        super().__init__()
        self.cdf = cdf
        self.output_shape = tuple(output_shape)

    def __call__(self, num_photon):
        u = torch.rand(num_photon)
        sampled_idx = torch.searchsorted(torch.tensor(self.cdf), u)

        output = torch.zeros(self.output_shape)
        unique_idx, counts = torch.unique(sampled_idx, return_counts=True)
        output.view(-1)[unique_idx] = counts.float()

        output = torch.nn.functional.pad(
            output,
            (2560, 16000 - output.shape[1] - 2560),
            mode="constant",
            value=0
        )
        return output
    
    def batch_sample(self, num_photon, num_batch):
        return torch.stack([self(num_photon) for _ in range(num_batch)])

data = np.load("data/lightLUT_Mod0_06052024_32.1.16_time_dist_cdf.npz")
mod0_sampler = TimingDistributionSampler(**data)