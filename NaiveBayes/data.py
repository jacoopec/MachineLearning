import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate 2 clusters with 2 features (X: shape (n_samples, 2), y: labels 0/1)
# Adjust n_samples, cluster_std for more/less overlap
n_samples = 300  # Total points
centers = [[-1, -1], [2, 2]]  # Cluster centers for diagonal separation
cluster_std = 1.0  # Standard deviation for overlap

X, y = make_blobs(
    n_samples=n_samples,
    centers=centers,
    n_features=2,
    cluster_std=cluster_std,
    random_state=80  # For reproducibility
)

# Optional: Plot to visualize (like your scatter plot)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', alpha=0.7)
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Now X is your feature matrix, y is the target label vector
print("Features shape:", X.shape)  # e.g., (300, 2)
print("Labels shape:", y.shape)    # e.g., (300,)
print("Sample data:\n", X[:5])
print("Sample labels:", y[:5])