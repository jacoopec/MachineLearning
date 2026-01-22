import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

# Custom KNN regressor
def knn_regressor(X_train, y_train, X_test, k=2):
    predictions = []
    neighbor_lists = []
    for x in X_test:
        # Compute distances to all training points
        dists = [distance.euclidean(x, x_train) for x_train in X_train]
        neighbors_idx = np.argsort(dists)[:k]
        pred = np.mean([y_train[i] for i in neighbors_idx])
        predictions.append(pred)
        neighbor_lists.append(neighbors_idx)
    return np.array(predictions), neighbor_lists

# Sample 2D feature inputs (x1, x2) and 1D target outputs (y)
X_train = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 3.0],
    [6.0, 5.0]
])
y_train = np.array([2.5, 2.0, 3.5, 6.0])

# Test points to predict
X_test = np.array([
    [2.5, 2.0],
    [5.0, 4.0]
])

# Run KNN regression
y_pred, neighbor_indices = knn_regressor(X_train, y_train, X_test, k=2)

# ----- 3D Visualization -----
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot training points
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='black', s=80, label="Training Points")
for i in range(len(X_train)):
    ax.text(X_train[i, 0], X_train[i, 1], y_train[i], f"{y_train[i]}", fontsize=9)

# Plot test points and lines to neighbors
for i, test_point in enumerate(X_test):
    ax.scatter(test_point[0], test_point[1], y_pred[i], color='orange', s=100, marker='X',
               label=f"Test {i} (pred={round(y_pred[i], 2)})" if i == 0 else "")
    for idx in neighbor_indices[i]:
        x_vals = [test_point[0], X_train[idx, 0]]
        y_vals = [test_point[1], X_train[idx, 1]]
        z_vals = [y_pred[i], y_train[idx]]
        ax.plot(x_vals, y_vals, z_vals, 'k--', alpha=0.5)

# Labels
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Target / Prediction")
ax.set_title("3D KNN Regression Visualization (k=2)")
ax.legend()
plt.tight_layout()

# Show the plot
plt.show()
