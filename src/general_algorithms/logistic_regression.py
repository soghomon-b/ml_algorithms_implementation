from abc import ABC, abstractmethod
import numpy as np

class LogisticRegression(ABC):
    def __init__(self):
        self.W = None  # (d+1,)

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        n, d = X.shape
        X_b = np.hstack([np.ones((n, 1)), X])
        z = X_b @ self.W
        return self._sigmoid(z)

    @staticmethod
    def _sigmoid(z):
        # numerically stable
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionWithGradientDescent(LogisticRegression):
    def __init__(self, lr=0.01, num_epochs=100):
        super().__init__()
        self.lr = lr
        self.num_epochs = num_epochs

    def fit(self, X, y):
        n, d = X.shape
        X_b = np.hstack([np.ones((n, 1)), X])   # (n, d+1)
        y = y.astype(float).reshape(-1)
        if self.W is None:
            self.W = np.zeros(d + 1)

        for _ in range(self.num_epochs):
            z = X_b @ self.W                    # (n,)
            p = self._sigmoid(z)                # (n,)
            grad = (X_b.T @ (p - y)) / n        # (d+1,)
            self.W -= self.lr * grad            # descent step


class LogisticRegressionWithStochasticGradientDescent(LogisticRegression):
    def __init__(self, lr=0.01, num_epochs=10, shuffle=True):
        super().__init__()
        self.lr = lr
        self.num_epochs = num_epochs
        self.shuffle = shuffle

    def fit(self, X, y):
        n, d = X.shape
        X_b = np.hstack([np.ones((n, 1)), X])   # (n, d+1)
        y = y.astype(float).reshape(-1)
        if self.W is None:
            self.W = np.zeros(d + 1)

        idx = np.arange(n)
        for _ in range(self.num_epochs):
            if self.shuffle:
                np.random.shuffle(idx)
            for i in idx:
                xi = X_b[i]                     # (d+1,)
                yi = y[i]                       # scalar
                z = xi @ self.W                 # scalar
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
                grad = (p - yi) * xi            # (d+1,)
                self.W -= self.lr * grad        # per-sample update (no /n)
