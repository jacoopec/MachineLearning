import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Custom KNN regressor function (same as before)
def knn_regressor(X_train, y_train, X_test, k=2):
    predictions = []
    neighbor_lists = []
    for x in X_test:
        # Calculate Euclidean distances
        dists = [distance.euclidean(x, x_train) for x_train in X_train]
        # Find k nearest neighbors
        neighbors_idx = np.argsort(dists)[:k]
        # Predict by averaging their y values
        pred = np.mean([y_train[i] for i in neighbors_idx])
        predictions.append(pred)
        neighbor_lists.append(neighbors_idx)
    return np.array(predictions), neighbor_lists


# ---- DATA EXAMPLE IN 2D FEATURE SPACE ----

# Training data: 2D features
X_train = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 3.0],
    [3.5, 4.0],
    [5.5, 4.5],
    [6.0, 5.0]
])

# Corresponding target values
y_train = np.array([2.5, 2.0, 3.5,5.0,5.5, 6.0])

# Test data: 2D features
X_test = np.array([
    [2.5, 2.0],
    [5.0, 4.0]
])

# Predict using KNN with k=2
y_pred, neighbor_indices = knn_regressor(X_train, y_train, X_test, k=2)

# Output predictions
print("Predictions:", y_pred)


plt.figure(figsize=(8, 6))

# Plot training points
for i, (point, label) in enumerate(zip(X_train, y_train)):
    plt.scatter(*point, color='black', s=100, label=f"Train {i}" if i == 0 else "")
    plt.text(point[0]+0.1, point[1], f"y={label}", fontsize=9)

# Plot test points and neighbor connections
for i, test_point in enumerate(X_test):
    plt.scatter(*test_point, color='orange', s=150, marker='X',
                label=f"Test {i} (pred={round(y_pred[i], 2)})")
    for idx in neighbor_indices[i]:
        neighbor = X_train[idx]
        plt.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]], 'k--', alpha=0.5)

# Labels and layout
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Regression Visualization (k=2)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
