import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param  = lambda_param
        self.epochs        = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # SVM usa etichette {-1, +1}
        y = np.where(y <= 0, -1, 1)

        # Inizializzazione
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                condition = y[i] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Solo regolarizzazione
                    dw = 2 * self.lambda_param * self.w
                    self.w -= self.learning_rate * dw
                else:
                    # Regolarizzazione + hinge loss
                    dw = (2 * self.lambda_param * self.w - y[i] * x_i)

                    db = -y[i]

                    self.w -= self.learning_rate * dw
                    self.b -= self.learning_rate * db

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b

        return np.sign(linear_output)


# ---------------------------------------------------
# ESEMPIO
# ---------------------------------------------------

X = np.array([[1, 2],[2, 3],[2, 1],[3, 2],[6, 5],[7, 7],[8, 6],[7, 5]], dtype=float)

y = np.array([-1,-1,-1,-1,1,1,1,1])


# Training
model = SVM(
    learning_rate=0.001,
    lambda_param=0.01,
    epochs=2000
)

model.fit(X, y)


# Prediction
predictions = model.predict(X)

print("Predictions:")
print(predictions)

print("\nWeights:")
print(model.w)

print("\nBias:")
print(model.b)


# Nuovo punto
new_point = np.array([
    [5, 4]
])

prediction = model.predict(new_point)

print("\nNew point:", new_point)
print("Prediction:", prediction)