
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic 3D data
np.random.seed(42)
mean = [0, 0, 0]
cov = [[3, 1, 1], [1, 2, 0.5], [1, 0.5, 1]]
X = np.random.multivariate_normal(mean, cov, size=300)

# Apply PCA (reduce to 2D for projection and visual clarity)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_projected = pca.inverse_transform(X_pca)

# Get full 3D PCA (to extract all 3 principal components)
pca_full = PCA(n_components=3)
pca_full.fit(X)
eigenvectors = pca_full.components_
mean_point = pca_full.mean_

# Create meshgrid for PCA plane
grid_size = 10
grid_range = np.linspace(-3, 3, grid_size)
xx, yy = np.meshgrid(grid_range, grid_range)
zz = np.zeros_like(xx)
grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_3d = pca.inverse_transform(grid_2d)
xx_p, yy_p, zz_p = grid_3d[:, 0].reshape(xx.shape), grid_3d[:, 1].reshape(yy.shape), grid_3d[:, 2].reshape(zz.shape)

# 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot original and projected data
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.2, label='Original Data')
ax.scatter(X_projected[:, 0], X_projected[:, 1], X_projected[:, 2], color='red', alpha=0.6, label='PCA Projection')

# Plot PCA plane
ax.plot_surface(xx_p, yy_p, zz_p, color='yellow', alpha=0.3)

# Plot eigenvector directions
for i in range(3):
    vec = eigenvectors[i] * 5  # scale
    ax.quiver(mean_point[0], mean_point[1], mean_point[2],
              vec[0], vec[1], vec[2], color='black', linewidth=2,
              label=f'PC{i+1}' if i == 0 else None)

ax.set_title("PCA in 3D with Eigenvector Directions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()
plt.show()
