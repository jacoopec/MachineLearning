import numpy as np
import matplotlib.pyplot as plt

# The model naturally learns transitions between class regions based on where x lies.


# Stable softmax
def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# Data: 1D feature, 4 classes
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Parameters
num_classes = 4
w = np.zeros(num_classes)  # One weight per class
b = np.zeros(num_classes)  # One bias per class
lr = 0.1
epochs = 1000
loss_history = []

# Training loop
for epoch in range(epochs):
    total_loss = 0
    grad_w = np.zeros(num_classes)
    grad_b = np.zeros(num_classes)

    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        z = w * x_i + b
        probs = softmax(z)

        loss = -np.log(probs[y_i])
        total_loss += loss

        y_onehot = np.zeros(num_classes)
        y_onehot[y_i] = 1

        grad = probs - y_onehot
        grad_w += grad * x_i
        grad_b += grad

    w -= lr * grad_w / len(X)
    b -= lr * grad_b / len(X)
    loss_history.append(total_loss / len(X))

# Predict and plot
x_test = np.linspace(0, 13, 300)
probs_all = np.zeros((len(x_test), num_classes))

for i, x in enumerate(x_test):
    z = w * x + b
    probs_all[i] = softmax(z)

# Plot probabilities
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for class_idx in range(num_classes):
    plt.plot(x_test, probs_all[:, class_idx], label=f'P(class {class_idx})')
plt.scatter(X, y, c=y, cmap='tab10', edgecolor='k', label='Training data')
plt.title('4-Class Softmax Probabilities (1D)')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
