import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate synthetic linear data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X + 5 + np.random.randn(100, 1)  # true slope=3, intercept=5

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define a 1-layer neural network (just linear regression)
model = nn.Linear(in_features=1, out_features=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1):
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
    predicted = model(X_tensor).detach().numpy()
    plt.scatter(X, y, label="Data")
    plt.plot(X, predicted, color='red', label="Fitted Line")
    plt.title("Neural Network Solving Linear Regression")
    plt.legend()
    plt.show()

# Plot results

