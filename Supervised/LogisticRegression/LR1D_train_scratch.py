import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss
def compute_loss(y, y_pred):
    eps = 1e-10  # to prevent log(0)
    return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

# Training data
X = np.array([1, 1.5, 2, 2.5, 3,  3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,8,8.5, 9, 9.5, 10])
y = np.array([0, 0,   0,   0,  0,  0,  0,  0,  0,  1,   1, 1, 1,1, 1, 1,1,1,1])

# Parameters
w = 0.0  # weight
b = 0.0  # bias
lr = 0.1
epochs = 10000
loss_history =[]

# Training loop
for epoch in range(epochs):
    z = w * X + b
    y_pred = sigmoid(z)
    
    # Gradients
    dw = np.dot((y_pred - y), X) / len(X)
    db = np.sum(y_pred - y) / len(X)
    
    # Update parameters
    w -= lr * dw
    b -= lr * db
    
    if epoch % 100 == 0:
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)
        # print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Plotting
x_test = np.linspace(0, 11, 300)
y_test = sigmoid(w * x_test + b)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(x_test, y_test, color='red', label='Sigmoid curve')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision boundary')
plt.xlabel('X')
plt.ylabel('Predicted Probability')
plt.title('Manual 1D Logistic Regression')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
