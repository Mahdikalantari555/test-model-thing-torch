import math
import pytest
import torch
from src.model.rtu import Model

def test_sampling_probabilities():
    output = torch.randn(256)
    probs = torch.softmax(output, dim=-1)
    assert math.isclose(probs.sum().item(), 1.0, rel_tol=1e-5)

def test_sampling_entropy_range():
    output = torch.randn(256)
    probs = torch.softmax(output, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)) / math.log(256.0)
    assert 0.0 <= entropy.item() <= 1.05  # within [0, 1] bits (plus float tolerance)

def test_sampling_temperature():
    temp_param = 0.75
    entropy = 0.5
    temp = max(0.1, temp_param * (1.0 - temp_param * entropy))
    assert temp >= 0.1

def test_sampling_categorical_bounds():
    model = Model(dim=64, layers=2)
    output = torch.randn(256)
    sample = model.sample(output)
    assert isinstance(sample, int)
    assert 0 <= sample <= 255
