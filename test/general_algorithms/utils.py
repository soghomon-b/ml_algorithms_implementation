import torch
import pytest
from torch import nn
import numpy as np

def make_toy_binary(n=200, d=8, seed=0, margin=2.0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w_true = torch.randn(d, 1, generator=g)
    logits = X @ w_true + margin * torch.sign(torch.randn(n, 1, generator=g))
    y = (logits > 0).float()
    return X, y



def sgd_expected_after_one_epoch(X, y, lr, use_mean=False, w0=None):
    n, d = X.shape
    X_b = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1) if w0 is None else w0.copy()
    scale = (1.0 / n) if use_mean else 1.0  # match your code
    for i in range(n):
        xi = X_b[i]                 # shape (d+1,)
        yi = y[i]
        err = xi @ w - yi
        grad_i = scale * err * xi   # per-sample gradient for MSE
        w -= lr * grad_i
    return w