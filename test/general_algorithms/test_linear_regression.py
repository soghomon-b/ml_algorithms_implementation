import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.general_algorithms.linear_regression import LinearRegression, LinearRegressionWithLeastSquares, LinearRegressionWithGradientDescent

# a dummy subclass since LinearRegression is abstract
class DummyLinearRegression(LinearRegression):
    def fit(self, X, y):
        pass  # not needed for this test


def test_predict():
    # Arrange: simple dataset (n=2, d=1)
    X = np.array([[1.0],
                  [2.0]])
    model = DummyLinearRegression()

    # weights: bias = 2, slope = 3
    model.w = np.array([2.0, 3.0])   # (d+1,)

    # Act
    y_pred = model.predict(X)

    # Assert
    expected = np.array([2 + 3*1.0, 2 + 3*2.0])  
    np.testing.assert_allclose(y_pred, expected)
    assert y_pred.shape == (2,)

def test_fit_LinearRegressionWithLeastSquares():
    X = np.array([[1.0],
                  [2.0]])
    y = np.array([1.0, 0.0])
    model = LinearRegressionWithLeastSquares()

    model.fit(X, y)

    # Expected from normal equations with X_b = [[1,1],[1,2]]
    expected_w = np.array([2.0, -1.0])   
    np.testing.assert_allclose(model.W, expected_w)


def test_linear_regression_gd_one_step():
    X = np.array([[1.0], [2.0]], dtype=float)
    y = np.array([1.0, 0.0], dtype=float)
    n = X.shape[0]
    X_b = np.hstack([np.ones((n,1)), X])

    model = LinearRegressionWithGradientDescent()
    model.lr = 0.01
    model.num_epochs = 1  # make sure your code loops range(self.num_epochs)
    model.fit(X, y)

    expected_w = (2 * model.lr / n) * (X_b.T @ y)

    np.testing.assert_allclose(np.array(model.W, float), expected_w, rtol=1e-12, atol=1e-12)



def test_linear_regression_gd_converges_to_ls():
    X = np.array([[1.0],[2.0]], dtype=float)
    y = np.array([1.0, 0.0], dtype=float)
    n = X.shape[0]
    X_b = np.hstack([np.ones((n,1)), X])

    # Closed-form target
    w_star, *_ = np.linalg.lstsq(X_b, y, rcond=None)

    model = LinearRegressionWithGradientDescent()
    model.lr = 0.1        # pick a stable lr
    model.num_epochs = 2000
    model.fit(X, y)

    np.testing.assert_allclose(np.array(model.W, float), w_star, rtol=1e-3, atol=1e-4)



    









