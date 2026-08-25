import numpy as np
import matplotlib.pyplot as plt

# Softmax function
def softmax(z):
    z = z - np.max(z, axis=0)  # stability for each column
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0)

# Data: 3 classes in 2D
X = np.array([
    [4, 2], [1.5, 1.8], [2, 1.5], [2.5,3.5] , [0.8, 2.2], # class 0
    [4, 4], [4.5, 3.8], [5, 4.5], [5.6,4.7], [7,4.0], # class 1
    [6, 2.5], [8.5, 2], [9, 1.5] , [7.5,1.8] , [10.3,1.4]    # class 2
])
y = np.array([0, 0, 0,0,0, 1,1,1, 1, 1, 2,2,2, 2, 2])
num_classes = 3
num_features = 2

# One-hot encode targets
y_onehot = np.eye(num_classes)[y]  # shape: (9, 3)

# Parameters
W = np.zeros((num_classes, num_features))  # weights (3 x 2)
b = np.zeros((num_classes, 1))             # biases (3 x 1)
lr = 0.1
epochs = 1000
loss_history = []

# Training loop
for epoch in range(epochs):
    X_T = X.T  # shape: (2 x 9)
    Z = np.dot(W, X_T) + b  # shape: (3 x 9)
    P = softmax(Z)          # shape: (3 x 9)

    loss = -np.mean(np.log(P[y, np.arange(len(y))]))
    loss_history.append(loss)

    # Gradient
    dZ = P - y_onehot.T             # shape: (3 x 9)
    dW = np.dot(dZ, X) / len(X)     # shape: (3 x 2)
    db = np.mean(dZ, axis=1, keepdims=True)  # shape: (3 x 1)

    # Update
    W -= lr * dW
    b -= lr * db

# 🖼️ Plot decision regions
def predict(X_input):
    Z = np.dot(W, X_input.T) + b
    return np.argmax(softmax(Z), axis=0)

# Grid for decision surface
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = predict(grid).reshape(xx.shape)

plt.figure(figsize=(12, 5))

# Plot decision boundary
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=80)
plt.title("2D Softmax Classifier — Decision Regions")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title("Cross-Entropy Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()
