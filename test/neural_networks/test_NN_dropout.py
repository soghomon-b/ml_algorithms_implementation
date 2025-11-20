import sys
import os

import pytest
import torch
import torch.nn as nn

# Allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.neural_networks.simple_NNs import Model, NeuralNetworkDropout


@pytest.fixture
def model():
    # in_dim and hidden can be anything > 0
    return NeuralNetworkDropout(in_dim=4, hidden=8, p=0.5)


def test_is_subclass_of_model(model):
    # Should inherit from your base Model AND nn.Module
    assert isinstance(model, Model)
    assert isinstance(model, nn.Module)


def test_layer_dimensions(model):
    # Check the internal layers are wired correctly
    assert isinstance(model.linear, nn.Linear)
    assert model.linear.in_features == 4
    assert model.linear.out_features == 8

    assert isinstance(model.relu, nn.ReLU)
    assert isinstance(model.dropout, nn.Dropout)
    assert pytest.approx(model.dropout.p) == 0.5

    assert isinstance(model.output_layer, nn.Linear)
    assert model.output_layer.in_features == 8
    assert model.output_layer.out_features == 1


def test_forward_shape_and_requires_grad(model):
    x = torch.ones((5, 4), requires_grad=True)  # (N, in_dim)
    model.train()  # training mode: dropout active
    logits = model(x)

    # shape (N, 1)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (5, 1)

    # forward should participate in autograd by default
    assert logits.requires_grad is True


def test_predict_proba_and_predict_eval_mode(model):
    # In eval mode, dropout is disabled (deterministic)
    model.eval()

    x = torch.randn(3, 4)
    p = model.predict_proba(x)
    y = model.predict(x, threshold=0.5)

    # predict_proba checks
    assert isinstance(p, torch.Tensor)
    assert p.shape == (3, 1)
    assert (p >= 0.0).all() and (p <= 1.0).all()  # sigmoid outputs in [0, 1]
    assert p.requires_grad is False  # because of @torch.no_grad

    # predict checks
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 1)
    assert y.dtype == torch.int64
    # all outputs are 0 or 1
    assert set(y.view(-1).tolist()) <= {0, 1}


def test_dropout_effect_training_mode():
    # Make randomness visible by using train() mode
    model = NeuralNetworkDropout(in_dim=4, hidden=8, p=0.7)
    model.train()

    x = torch.randn(10, 4)

    outputs = [model(x) for _ in range(5)]

    # At least one pair of outputs should differ because of dropout.
    # (Extremely small chance all are identical, but practically zero.)
    all_equal = all(torch.allclose(outputs[0], out) for out in outputs[1:])
    assert not all_equal


def test_dropout_no_effect_when_p_zero():
    # With p=0, dropout should do nothing; outputs should be deterministic even in train mode
    model = NeuralNetworkDropout(in_dim=4, hidden=8, p=0.0)
    model.train()

    x = torch.randn(10, 4)
    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1, out2)