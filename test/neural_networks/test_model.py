import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.neural_networks.simple_NNs import Model


class DumbModel(Model):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) → logits: (N, 1) all zeros
        n = x.shape[0]
        return torch.zeros((n, 1), dtype=torch.float32)


@pytest.fixture
def model():
    return DumbModel()


def test_init(model):
    assert isinstance(model, nn.Module)


def test_predict_proba_shape_and_values(model):
    x = torch.ones((3, 2))
    p = model.predict_proba(x)

    # returns a tensor
    assert isinstance(p, torch.Tensor)

    # shape (N, 1)
    assert p.shape == (3, 1)

    # sigmoid(0) = 0.5 everywhere
    expected = torch.full((3, 1), 0.5)
    assert torch.allclose(p, expected)

    # no grad because of @torch.no_grad
    assert p.requires_grad is False


def test_predict_default_threshold(model):
    x = torch.ones((3, 2))
    y = model.predict(x)  # threshold=0.5

    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.int64
    assert y.shape == (3, 1)

    # sigmoid(0) = 0.5 → >= 0.5 → 1
    expected = torch.ones((3, 1), dtype=torch.int64)
    assert torch.equal(y, expected)


def test_predict_higher_threshold(model):
    x = torch.ones((3, 2))
    y = model.predict(x, threshold=0.7)

    # 0.5 >= 0.7 → 0
    expected = torch.zeros((3, 1), dtype=torch.int64)
    assert torch.equal(y, expected)
