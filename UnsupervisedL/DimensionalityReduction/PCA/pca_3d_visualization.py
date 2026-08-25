
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic 3D data (Gaussian blobs)
np.random.seed(42)
mean = [0, 0, 0]
cov = [[3, 1, 1], [1, 2, 0.5], [1, 0.5, 1]]
X = np.random.multivariate_normal(mean, cov, size=300)

# Apply PCA to reduce from 3D to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Project back to 3D space for visualization
X_projected = pca.inverse_transform(X_pca)

# Generate a PCA plane for visualization
grid_size = 10
grid_range = np.linspace(-3, 3, grid_size)
xx, yy = np.meshgrid(grid_range, grid_range)
zz = np.zeros_like(xx)
grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_3d = pca.inverse_transform(grid_2d)
xx_p, yy_p, zz_p = grid_3d[:, 0].reshape(xx.shape), grid_3d[:, 1].reshape(yy.shape), grid_3d[:, 2].reshape(zz.shape)

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot original data
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.2, label='Original Data')

# Plot projected data
ax.scatter(X_projected[:, 0], X_projected[:, 1], X_projected[:, 2], color='red', alpha=0.6, label='PCA Projection')

# Plot PCA plane
ax.plot_surface(xx_p, yy_p, zz_p, color='yellow', alpha=0.3)

ax.set_title("PCA in 3D: Projection onto 2D Subspace")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()
plt.show()
