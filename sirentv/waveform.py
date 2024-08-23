import math
from functools import partial
from typing import Callable, Dict, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from larndsim.consts import light


def print_grad(name):
    def hook(grad):
        print(f"Gradient for {name}: {grad}")
    return hook

T = TypeVar("T")
class Config(Dict[str, T]):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name: str) -> T:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: T) -> None:
        if isinstance(value, str):
            value = eval(value, {}, {"uniform": np.random.uniform})
        self[name] = value

class BatchedLightSimulation(nn.Module):
    def __init__(self, cfg=os.path.join(os.path.dirname(__file__), "../templates/waveform_sim.yaml"), verbose=False):
        super().__init__()

        if isinstance(cfg, str):
            cfg = yaml.safe_load(open(cfg,'r').read())
            print(cfg)
            cfg = Config(**cfg)

        self.cfg = cfg

        # Load in and transform parameters
        def logit(x):
            return np.log(x / (1 - x))

        def create_parameter(value, nominal):
            return nn.Parameter(torch.tensor(value / nominal, dtype=torch.float32))

        # Calculate and store nominal values
        self.nominal_values = {
            'singlet_fraction_logit': logit(cfg.NOMINAL_SINGLET_FRACTION),
            'log_tau_s': np.log10(cfg.NOMINAL_TAU_S),
            'log_tau_t': np.log10(cfg.NOMINAL_TAU_T),
            'log_light_oscillation_period': np.log10(cfg.NOMINAL_LIGHT_OSCILLATION_PERIOD),
            'log_light_response_time': np.log10(cfg.NOMINAL_LIGHT_RESPONSE_TIME),
            'light_gain': cfg.NOMINAL_LIGHT_GAIN
        }

        # Calculate current values
        current_values = {
            'singlet_fraction_logit': logit(cfg.SINGLET_FRACTION),
            'log_tau_s': np.log10(cfg.TAU_S),
            'log_tau_t': np.log10(cfg.TAU_T),
            'log_light_oscillation_period': np.log10(cfg.LIGHT_OSCILLATION_PERIOD),
            'log_light_response_time': np.log10(cfg.LIGHT_RESPONSE_TIME),
            'light_gain': cfg.LIGHT_GAIN
        }

        # Create parameters
        for name in self.nominal_values.keys():
            setattr(self, name, create_parameter(current_values[name], self.nominal_values[name]))
            setattr(self, 'nominal_' + name, self.nominal_values[name])

        # Constants
        self.light_tick_size = light.LIGHT_TICK_SIZE
        self.light_window = [1, 100]

        self.conv_ticks = math.ceil(
            (self.light_window[1] - self.light_window[0]) / self.light_tick_size
        )

        self.time_ticks = torch.arange(self.conv_ticks)

        self.k = 100


        if verbose:
            self.register_grad_hook()

    def to(self, device):
        self.time_ticks = self.time_ticks.to(device)
        for param in ['log_light_oscillation_period', 'log_light_response_time', 'light_gain']:
            if not isinstance(getattr(self, param), nn.Parameter):
                setattr(self, param, getattr(self, param).to(device))
        super().to(device)
        return self
    
    @property
    def device(self):
        return next(self.parameters()).device

    def register_grad_hook(self):
        for name, p in self.named_parameters():
            p.register_hook(print_grad(name))

    def scintillation_model(self, time_tick, relax_cut=True) -> torch.Tensor:
        """
        Calculates the fraction of scintillation photons emitted
        during time interval `time_tick` to `time_tick + 1`

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            relax_cut (bool): whether to apply the relaxing cut for differentiability

        Returns:
            torch.Tensor: fraction of scintillation photons
        """

        singlet_fraction = torch.sigmoid(self.singlet_fraction_logit * self.nominal_singlet_fraction_logit)
        tau_s = torch.pow(10, self.log_tau_s * self.nominal_log_tau_s)
        tau_t = torch.pow(10, self.log_tau_t * self.nominal_log_tau_t)
        t = time_tick * self.light_tick_size

        p1 = (
            singlet_fraction
            * torch.exp(-t / tau_s)
            * (1 - torch.exp(-self.light_tick_size / tau_s))
        )
        p3 = (
            (1 - singlet_fraction)
            * torch.exp(-t / tau_t)
            * (1 - torch.exp(-self.light_tick_size / tau_t))
        )
        
        if relax_cut:
            return (p1 + p3) / (1 + torch.exp(-self.k * t))

        return (p1 + p3) * (t >= 0).float()
    

    def sipm_response_model(self, time_tick, relax_cut=True) -> torch.Tensor:
        """
        Calculates the SiPM response from a PE at `time_tick` relative to the PE time

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            relax_cut (bool): whether to apply the relaxing cut for differentiability

        Returns:
            torch.Tensor: response
        """

        t = time_tick * self.light_tick_size
        light_oscillation_period = torch.pow(10, self.log_light_oscillation_period * self.nominal_log_light_oscillation_period)
        light_response_time = torch.pow(10, self.log_light_response_time * self.nominal_log_light_response_time)

        impulse = torch.exp(-t / light_response_time) * torch.sin(
            t / light_oscillation_period
        )
        if relax_cut:
            impulse = impulse / (1 + torch.exp(-self.k * time_tick))
        else:
            impulse = impulse * (time_tick >= 0).float()

        impulse /= light_oscillation_period * light_response_time**2
        impulse *= light_oscillation_period**2 + light_response_time**2
        return impulse * self.light_tick_size

    
    def fft_conv(self, light_sample_inc: torch.Tensor, model: Callable) -> torch.Tensor:
        """
        Performs a Fast Fourier Transform (FFT) convolution on the input light sample.

        Args:
            light_sample_inc (torch.Tensor): Light incident on each detector.
                Shape: (ninput, ndet, ntick)
            model (callable): Function that generates the convolution kernel.

        Returns:
            torch.Tensor: Convolved light sample.
                Shape: (ninput, ndet, ntick)

        This method applies the following steps:
        1. Pads the input tensor
        2. Computes the convolution kernel using the provided model
        3. Performs FFT on both the input and the kernel
        4. Multiplies the FFTs in the frequency domain
        5. Performs inverse FFT to get the convolved result
        6. Reshapes and trims the output to match the input shape
        """
        ninput, ndet, ntick = light_sample_inc.shape

        # Pad the input tensor
        pad_size = self.conv_ticks - 1
        padded_input = F.pad(light_sample_inc, (0, pad_size))

        # Compute kernel
        scintillation_kernel = model(self.time_ticks)
        kernel_fft = torch.fft.rfft(scintillation_kernel, n=ntick + pad_size)

        # Reshape for batched FFT convolution
        padded_input = padded_input.reshape(ninput * ndet, ntick + pad_size)

        # Perform FFT convolution
        input_fft = torch.fft.rfft(padded_input)
        output_fft = input_fft * kernel_fft.unsqueeze(0)
        output = torch.fft.irfft(output_fft, n=ntick + pad_size)

        # Reshape and trim the result to match the input shape
        output = output.reshape(ninput, ndet, -1)[:, :, :ntick]

        return output
    
    def conv(self, light_sample_inc: torch.Tensor, model: Callable) -> torch.Tensor:
        """
        Performs a grouped convolution on the input light sample.

        Args:
            light_sample_inc (torch.Tensor): Light incident on each detector.
                Shape: (ninput, ndet, ntick)
            model (callable): Function that generates the convolution kernel.

        Returns:
            torch.Tensor: Convolved light sample.
                Shape: (ninput, ndet, ntick)

        This method applies the following steps:
        1. Pads the input tensor
        2. Reshapes the input for grouped convolution
        3. Computes the convolution kernel using the provided model
        4. Performs grouped convolution using torch.nn.functional.conv1d
        5. Reshapes and trims the output to match the input shape
        """
        ninput, ndet, ntick = light_sample_inc.shape

        # Pad the input tensor
        padded_input = torch.nn.functional.pad(
            light_sample_inc, (self.conv_ticks - 1, 0)
        )

        # Reshape for grouped convolution
        padded_input = padded_input.view(1, ninput * ndet, -1)

        kernel = model(self.time_ticks).flip(0)

        # Create a separate kernel for each input and detector
        kernel = (
            kernel.unsqueeze(0).expand(ninput * ndet, -1).unsqueeze(1)
        )

        # Perform the convolution
        output = torch.nn.functional.conv1d(
            padded_input, kernel, groups=ninput * ndet
        )

        # Trim the result to match the input shape
        output = output.view(ninput, ndet, -1)[:, :, :ntick]

        return output

    def downsample_waveform(self, waveform, ns_per_tick=16):
        """
        Downsample the input waveform by summing over groups of ticks.
        This effectively compresses the waveform in the time dimension while preserving the total signal.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (ninput, ndet, ntick), where each tick corresponds to 1 ns.
            ns_per_tick (int, optional): Number of nanoseconds per tick in the downsampled waveform. Defaults to 16.

        Returns:
            torch.Tensor: Downsampled waveform of shape (ninput, ndet, ntick_down).
        """
        ninput, ndet, ntick = waveform.shape
        ntick_down = ntick // ns_per_tick
        downsample = waveform.view(ninput, ndet, ntick_down, ns_per_tick).sum(dim=3)
        return downsample

    def forward(self, timing_dist, differentiable=True):
        reshaped = False
        if timing_dist.ndim == 1:  # ndet=1, ninput=1; (ntick) -> (1, 1, ntick)
            timing_dist = timing_dist[None, None, :]
            reshaped = True
        elif timing_dist.ndim == 2:  # ndet>1, ninput=1; (ndet, ntick) -> (1, ndet, ntick)
            timing_dist = timing_dist[None, :, :]
            reshaped = True

        x = self.fft_conv(timing_dist, partial(self.scintillation_model, relax_cut=differentiable))
        x = self.fft_conv(x, partial(self.sipm_response_model, relax_cut=differentiable))
        x = self.light_gain * x
        x = self.downsample_waveform(x)

        if reshaped:
            return x.squeeze(0).squeeze(0)

        return x

class TimingDistributionSampler:
    def __init__(self, cdf: np.ndarray, output_shape: tuple):
        super().__init__()
        self.cdf = cdf
        self.output_shape = tuple(output_shape)

    def __call__(self, num_photon: int):
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
    
    def batch_sample(self, nphoton: int, nbatch: int) -> torch.Tensor:
        return torch.stack([self(nphoton) for _ in range(nbatch)])


data_path = os.path.join(os.path.dirname(__file__), "../data/lightLUT_Mod0_06052024_32.1.16_time_dist_cdf.npz")
data = np.load(data_path)
mod0_sampler = TimingDistributionSampler(**data)