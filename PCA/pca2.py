import numpy as np
import matplotlib.pyplot as plt

# Very simple 2D dataset
X = np.array([
    [2, 3],
    [4, 3],
    [8, 3],
    [3, 5],
    [4, 7],
    [5, 8],
    [6, 9]
])


# 1. Mean-center the data
mean = np.mean(X, axis=0)
X_centered = X - mean

# 2. Covariance matrix
cov = np.cov(X_centered.T)

# 3. Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)

# 4. Sort eigenvectors by importance
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]


fig, axes = plt.subplots(1, 2)
axes[0].scatter(X_centered[:, 0], X_centered[:, 1], color="blue")
axes[0].scatter(X[:, 0], X[:, 1], color="yellow")

origin = np.array([[0, 0]])

# Plot each principal component
for i in range(2):
    vec = eigenvectors[:, i]
    axes[0].arrow(0, 0, vec[0]*2, vec[1]*2,      # scale vectors for visibility
              color='red', width=0.02, 
              head_width=0.1, label=f"PC{i+1}")

plt.xlabel("x1 (centered)")
plt.ylabel("x2 (centered)")
plt.title("Simple PCA Example")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
