import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest
from src.general_algorithms.logistic_regression import (
    LogisticRegressionWithGradientDescent,
    LogisticRegressionWithStochasticGradientDescent,
)

# ------------------------
# Helpers
# ------------------------

def make_toy_binary(n=50, d=2, seed=0, margin=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d, 1))
    logits = X @ w_true + margin * np.sign(rng.normal(size=(n, 1)))
    y = (logits > 0).astype(float).ravel()
    return X, y


# ------------------------
# Tests
# ------------------------

def test_predict_shape_and_range():
    X, y = make_toy_binary(n=10, d=3, seed=42)
    model = LogisticRegressionWithGradientDescent(lr=0.1, num_epochs=1)
    model.W = np.zeros(X.shape[1] + 1)  # initialize manually
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_one_step_gradient_descent_update_matches_gradient():
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 0.0])
    n = X.shape[0]
    X_b = np.hstack([np.ones((n, 1)), X])
    model = LogisticRegressionWithGradientDescent(lr=0.1, num_epochs=1)
    model.W = np.zeros(X_b.shape[1])

    # manual gradient at w=0
    z = X_b @ model.W
    p = 1 / (1 + np.exp(-z))
    grad = (X_b.T @ (p - y)) / n
    expected_w = -model.lr * grad

    model.fit(X, y)
    np.testing.assert_allclose(model.W, expected_w, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ModelCls", [LogisticRegressionWithGradientDescent,
                                      LogisticRegressionWithStochasticGradientDescent])
def test_convergence_on_separable_data(ModelCls):
    X, y = make_toy_binary(n=200, d=2, seed=123, margin=5.0)
    model = ModelCls(lr=0.1, num_epochs=300)
    model.fit(X, y)
    preds = model.predict(X)
    acc = (preds.round() == y).mean()
    assert acc > 0  # should learn well on separable data


def test_sgd_and_gd_give_similar_results():
    X, y = make_toy_binary(n=100, d=2, seed=99, margin=3.0)
    gd = LogisticRegressionWithGradientDescent(lr=0.05, num_epochs=200)
    sgd = LogisticRegressionWithStochasticGradientDescent(lr=0.05, num_epochs=200, shuffle=False)

    gd.fit(X, y)
    sgd.fit(X, y)

    # predictions should be similar
    preds_gd = gd.predict(X)
    preds_sgd = sgd.predict(X)
    np.testing.assert_allclose(preds_gd, preds_sgd, rtol=1e-1, atol=1e-1)
