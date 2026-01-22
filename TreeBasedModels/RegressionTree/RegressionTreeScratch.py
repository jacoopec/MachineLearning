import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([0.3, 0.6, 0.65, 0.8, 0.95, 1.2, 1.23, 1.3, 1.45, 1.49,
              1.55, 1.7, 1.73, 1.79, 1.88, 1.98, 2.03, 2.05, 2.1])
Y = np.array([0.9, 1.2, 1.15, 1.3, 2.00, 19.3, 20.01, 20.02, 21.02, 22,
              20.7, 14.98, 15.05, 15.9, 15.2, 3.03, 2.69, 2.98, 3.1])

# Sort data
sorted_indices = np.argsort(X)
X = X[sorted_indices]
Y = Y[sorted_indices]

# Regression tree node class
class TreeNode:
    def __init__(self, prediction=None, split=None, left=None, right=None):
        self.prediction = prediction
        self.split = split
        self.left = left
        self.right = right

# Recursive tree builder
def build_tree(X, Y, depth, max_depth):
    if len(X) < 2 or depth == max_depth:
        return TreeNode(prediction=np.mean(Y))

    best_split = None
    best_error = float('inf')
    best_left = best_right = None

    for i in range(1, len(X)):
        split = (X[i-1] + X[i]) / 2
        left_mask = X <= split
        right_mask = X > split

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue

        Y_left, Y_right = Y[left_mask], Y[right_mask]
        mean_left, mean_right = np.mean(Y_left), np.mean(Y_right)

        error = np.sum((Y_left - mean_left)**2) + np.sum((Y_right - mean_right)**2)

        if error < best_error:
            best_error = error
            best_split = split
            best_left = (X[left_mask], Y[left_mask])
            best_right = (X[right_mask], Y[right_mask])

    if best_split is None:
        return TreeNode(prediction=np.mean(Y))

    left_subtree = build_tree(*best_left, depth + 1, max_depth)
    right_subtree = build_tree(*best_right, depth + 1, max_depth)
    return TreeNode(split=best_split, left=left_subtree, right=right_subtree)

# Prediction function
def predict_tree(tree, x):
    if tree.prediction is not None:
        return tree.prediction
    if x <= tree.split:
        return predict_tree(tree.left, x)
    else:
        return predict_tree(tree.right, x)

# Build the tree (change max_depth here)
max_depth = 3
tree = build_tree(X, Y, depth=0, max_depth=max_depth)

# Generate predictions
X_test = np.linspace(min(X), max(X), 500)
Y_pred = np.array([predict_tree(tree, x) for x in X_test])

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label=f'Regression Tree (depth={max_depth})')
plt.title(f'Regression Tree From Scratch (max_depth={max_depth})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()







# for split in split_candidates:
#     plt.axvline(x=split, color='purple', linestyle='--', linewidth=0.5)
