import numpy as np

class LinearRegression:

    def __init__(self, lr=0.01, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []  # To store loss values

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias

            # Loss computation (Mean Squared Error)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # Gradient computation
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
