import numpy as np

class LinearSVM:
    def __init__(self, lr=0.001, C=1.0, epochs=1000):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Convert labels to {-1, +1}
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Only regularization term contributes
                    dw = self.w
                    db = 0
                else:
                    # Hinge loss active
                    dw = self.w - self.C * y_[idx] * x_i
                    db = -self.C * y_[idx]

                # Gradient descent update
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
