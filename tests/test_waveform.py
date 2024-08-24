import pytest
import torch
import numpy as np
from sirentv.waveform import BatchedLightSimulation, TimingDistributionSampler, mod0_sampler

@pytest.fixture
def batched_light_simulation():
    return BatchedLightSimulation()

@pytest.fixture
def timing_distribution_sampler():
    return mod0_sampler

def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

def test_single_input(batched_light_simulation):
    # Test with a single input (1 detector, 1 time distribution)
    single_input = torch.rand(16000)
    output = batched_light_simulation(single_input)
    assert output.shape == (1000,), f"Expected shape (1000), got {output.shape}"

def test_multiple_detectors(batched_light_simulation):
    # Test with multiple detectors
    multi_detector_input = torch.rand(1, 10, 16000)
    output = batched_light_simulation(multi_detector_input)
    assert output.shape == (1, 10, 1000), f"Expected shape (1, 10, 1000), got {output.shape}"

    multi_detector_input = torch.rand(10, 16000)
    output = batched_light_simulation(multi_detector_input)
    assert output.shape == (10, 1000), f"Expected shape (10, 1000), got {output.shape}"

def test_batch_input(batched_light_simulation):
    # Test with a batch of inputs
    batch_input = torch.rand(5, 10, 16000)
    output = batched_light_simulation(batch_input)
    assert output.shape == (5, 10, 1000), f"Expected shape (5, 10, 1000), got {output.shape}"

def test_zero_input(batched_light_simulation):
    # Test with zero input
    zero_input = torch.zeros(2, 2, 16000)
    output = batched_light_simulation(zero_input)
    assert torch.allclose(output, torch.zeros(2, 2, 1000)), "Expected all zeros output"

def test_sampled_input(batched_light_simulation, timing_distribution_sampler):
    # Test with sampled input from TimingDistributionSampler
    sampled_input = timing_distribution_sampler(123)
    output = batched_light_simulation(sampled_input)
    assert output.shape == (sampled_input.shape[0], 1000), f"Expected shape {(sampled_input.shape[0], 1000)}, got {output.shape}"

def test_differentiability(batched_light_simulation):
    # Test differentiability
    input_tensor = torch.rand(1, 1, 16000, requires_grad=True)

    output = batched_light_simulation(input_tensor, relax_cut=False)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None, "Input should have gradients"

    input_tensor = torch.rand(1, 1, 16000, requires_grad=True)
    output = batched_light_simulation(input_tensor, relax_cut=False)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None, "Input should have gradients"

def test_large_batch(batched_light_simulation):
    # Test with a large batch
    large_batch = torch.rand(100, 5, 16000)
    output = batched_light_simulation(large_batch)
    assert output.shape == (100, 5, 1000), f"Expected shape (100, 5, 1000), got {output.shape}"
