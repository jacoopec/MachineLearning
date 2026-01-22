import numpy as np
import matplotlib.pyplot as plt


# Softmax generalizes sigmoid to 3 classes
# Even with just 1D input, the model can learn to separate multiple classes using linear scores.
# Each class gets its own bk  and wk
# Training uses cross-entropy loss with manual gradients.

# A notebook version?
# A 2D feature space (for more intuitive visual separation)?
# Or even train using PyTorch manually to compare?

# Softmax function (numerically stable)
def softmax(z):
    z = z - np.max(z)  # stabilize
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# Training data (1D, 3 classes)
X = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# Parameters
num_classes = 3
w = np.zeros(num_classes)
b = np.zeros(num_classes)
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

        # One-hot vector
        y_onehot = np.zeros(num_classes)
        y_onehot[y_i] = 1

        grad = probs - y_onehot
        grad_w += grad * x_i
        grad_b += grad

    # Update parameters
    w -= lr * grad_w / len(X)
    b -= lr * grad_b / len(X)
    loss_history.append(total_loss / len(X))

# Test softmax probabilities over x-range
x_test = np.linspace(0, 12, 300)
probs_all = np.zeros((len(x_test), num_classes))

for i, x in enumerate(x_test):
    z = w * x + b
    probs_all[i] = softmax(z)

# Plot probabilities
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for class_idx in range(num_classes):
    plt.plot(x_test, probs_all[:, class_idx], label=f'P(class {class_idx})')
plt.scatter(X, y, c=y, cmap='viridis', edgecolor='k', label='Training data')
plt.title('3-Class Softmax Probabilities')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
