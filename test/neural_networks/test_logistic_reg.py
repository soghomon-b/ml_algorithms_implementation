import sys
import os

import pytest
import torch
import torch.nn as nn

# Allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.neural_networks.simple_NNs import Model, LogisticRegressionNN


def make_zero_log_reg(in_dim: int) -> LogisticRegressionNN:
    """
    Helper: create a LogisticRegressionNN with all weights/bias = 0
    so logits = 0 and sigmoid(logits) = 0.5 deterministically.
    """
    model = LogisticRegressionNN(in_dim=in_dim)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    return model


@pytest.fixture
def model():
    # in_dim = 2 here, adjust if you need
    return make_zero_log_reg(in_dim=2)


def test_is_subclass_of_base_model(model):
    # LogisticRegressionNN should be a subclass of your base Model
    assert isinstance(model, Model)
    assert isinstance(model, nn.Module)


def test_linear_layer_dimensions(model):
    linear = model.linear
    assert isinstance(linear, nn.Linear)
    # in_features should match in_dim (=2 in the fixture above)
    assert linear.in_features == 2
    # out_features should be 1 for logistic regression
    assert linear.out_features == 1


def test_predict_proba_shape_and_values(model):
    # 3 samples, in_dim=2
    x = torch.ones((3, 2))
    p = model.predict_proba(x)

    # returns a tensor
    assert isinstance(p, torch.Tensor)

    # shape (N, 1)
    assert p.shape == (3, 1)

    # weights & bias are zero → logits = 0 → sigmoid(0) = 0.5
    expected = torch.full((3, 1), 0.5)
    assert torch.allclose(p, expected)

    # no gradients should be tracked because of @torch.no_grad
    assert p.requires_grad is False


def test_predict_default_threshold(model):
    x = torch.ones((3, 2))
    y = model.predict(x)  # default threshold = 0.5

    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.int64
    assert y.shape == (3, 1)

    # 0.5 >= 0.5 → 1
    expected = torch.ones((3, 1), dtype=torch.int64)
    assert torch.equal(y, expected)


def test_predict_higher_threshold(model):
    x = torch.ones((3, 2))
    y = model.predict(x, threshold=0.7)

    # 0.5 >= 0.7 → 0
    expected = torch.zeros((3, 1), dtype=torch.int64)
    assert torch.equal(y, expected)


def test_forward_requires_grad():
    # separate test to ensure normal forward uses gradients
    model = LogisticRegressionNN(in_dim=2)
    x = torch.ones((4, 2), requires_grad=True)
    logits = model(x)

    assert logits.shape == (4, 1)
    assert logits.requires_grad is True
