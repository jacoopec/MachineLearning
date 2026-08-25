import numpy as np
import matplotlib.pyplot as plt

# Softmax for 2-class (numerically stable version)
def softmax(z1, z2):
    z = np.array([z1, z2])
    z -= np.max(z)  # for numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# Training data
X = np.array([1, 2, 3, 7, 8, 9])
y = np.array([0, 0, 0, 1, 1, 1])

# Parameters for 2 classes: w0, b0 and w1, b1
w = np.array([0.0, 0.0])  # weights for class 0 and 1
b = np.array([0.0, 0.0])  # biases for class 0 and 1

lr = 0.1
epochs = 1000
loss_history = []

# Training loop
for epoch in range(epochs):
    total_loss = 0
    grad_w = np.zeros(2)
    grad_b = np.zeros(2)

    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        # Raw scores
        z0 = w[0] * x_i + b[0]
        z1 = w[1] * x_i + b[1]

        # Softmax probabilities
        probs = softmax(z0, z1)

        # Cross-entropy loss for correct class
        loss = -np.log(probs[y_i])
        total_loss += loss

        # One-hot encoding of target
        y_onehot = np.array([1, 0]) if y_i == 0 else np.array([0, 1])

        # Gradient for each class
        grad = probs - y_onehot
        grad_w += grad * x_i
        grad_b += grad

    # Update weights and biases
    w -= lr * grad_w / len(X)
    b -= lr * grad_b / len(X)
    loss_history.append(total_loss / len(X))

# Plot softmax probabilities over input range
x_test = np.linspace(0, 10, 300)
probs_class0 = []
probs_class1 = []

for x in x_test:
    z0 = w[0] * x + b[0]
    z1 = w[1] * x + b[1]
    probs = softmax(z0, z1)
    probs_class0.append(probs[0])
    probs_class1.append(probs[1])

# Plot the softmax decision boundary
plt.figure(figsize=(12, 5))

# Plot probabilities
plt.subplot(1, 2, 1)
plt.plot(x_test, probs_class0, label='P(class 0)', color='blue')
plt.plot(x_test, probs_class1, label='P(class 1)', color='red')
plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k', label='Data')
plt.axhline(0.5, color='gray', linestyle='--')
plt.title('Softmax 2-Class Probabilities')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

# Plot loss over epochs
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
