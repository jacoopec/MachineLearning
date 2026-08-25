import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(0)
num_samples = 100

# Class 0
x0 = np.random.multivariate_normal([2, 2], [[1, 0.75],[0.75, 1]], num_samples)
y0 = np.zeros(num_samples)

# Class 1
x1 = np.random.multivariate_normal([4, 4], [[1, 0.75],[0.75, 1]], num_samples)
y1 = np.ones(num_samples)

# Combine
X = np.vstack((x0, x1))
y = np.hstack((y0, y1))

# Step 2: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Loss (binary cross-entropy)
def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))

# Step 4: Training
def train_logistic_regression(X, y, lr=0.1, n_iters=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    losses = []

    for _ in range(n_iters):
        linear = np.dot(X, w) + b
        y_hat = sigmoid(linear)

        # Gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
        db = (1 / n_samples) * np.sum(y_hat - y)

        # Update
        w -= lr * dw
        b -= lr * db

        # Store loss
        losses.append(compute_loss(y, y_hat))

    return w, b, losses

# Step 5: Train model
w, b, losses = train_logistic_regression(X, y)

# Step 6: Plot decision boundary
def plot_decision_boundary(w, b, X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    x_values = [X[:, 0].min(), X[:, 0].max()]
    y_values = -(w[0] * np.array(x_values) + b) / w[1]
    plt.plot(x_values, y_values, label='Decision Boundary', color='black')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.show()

plot_decision_boundary(w, b, X, y)

# Step 7: Plot training loss
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()
