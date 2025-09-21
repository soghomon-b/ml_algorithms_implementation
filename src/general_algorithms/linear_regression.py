from abc import ABC
from abc import abstractmethod
import numpy as np

class LinearRegression(ABC):
    W = None
    @abstractmethod
    def fit(self, X,y):
        pass

    def predict(self, X):
        n, d = X.shape
        ones = np.ones((n, 1))
        X_b = np.hstack([ones, X])
        return X_b @ self.w

class LinearRegressionWithLeastSquares(LinearRegression):
    
    def fit(self,X,y):
        n, d = X.shape
        ones = np.ones((n, 1))        # shape (n,1)
        X_b = np.hstack([ones, X])    # shape (n, d+1)
        self.W = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

class LinearRegressionWithGradientDescent(LinearRegression):
    lr = 0.01
    num_epochs = 10
    def fit(self,X,y):
        n, d = X.shape
        ones = np.ones((n, 1))
        X_b = np.hstack([ones, X])   # add bias column

        # initialize weights
        self.W = np.zeros(d + 1)

        # run gradient descent
        for _ in range(self.num_epochs):
            y_pred = X_b @ self.W
            error = y_pred - y
            grad = (2/n) * X_b.T @ error
            self.W -= self.lr * grad

class LinearRegressionWithStochasticGradientDescent(LinearRegression):
    lr = 0.01
    num_epochs = 10
    def fit(self,X,y):
        n, d = X.shape
        ones = np.ones((n, 1))
        X_b = np.hstack([ones, X])   # add bias column

        # initialize weights
        self.w = np.zeros(d + 1)

        # run gradient descent
        idx = np.arange(n)
        for _ in range(self.num_epochs):
            if self.shuffle:
                np.random.shuffle(idx)
            for i in idx:
                xi = X_b[i]                        # shape (d+1,)
                yi = y[i]
                err = xi @ self.w - yi
                grad = 2.0 * err * xi             # ∇ for one sample
                self.w -= self.lr * grad


        


        



