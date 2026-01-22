import numpy as np
from sklearn.datasets import load_wine
import pandas as pd

# -------------------------------
# 1. Load dataset
# -------------------------------
# data = load_wine()
# X = data.data
# y = data.target

data = {
    "feature1": [1, 2, 2, 8, 9, 9, 1, 2, 3],
    "feature2": [9, 11, 2, 18, 9, 8, 4, 2, 3],
    "class":    [0, 0, 0, 1, 1, 1, 2, 2, 2]
}

X = np.column_stack((data["feature1"], data["feature2"]))
y = np.array(data["class"])


# number of samples, features
n, d = X.shape  
classes = np.unique(y)
k = len(classes)   # number of classes

# -------------------------------
# 2. Compute overall mean
# -------------------------------
mean_overall = np.mean(X, axis=0)
print(mean_overall)

# -------------------------------
# 3. Compute Within-class scatter SW
# -------------------------------
SW = np.zeros((d, d))

for c in classes:
    X_c = X[y == c]
    mean_c = np.mean(X_c, axis=0)
    print(X_c)
    SW += np.dot((X_c - mean_c).T, (X_c - mean_c))

# -------------------------------
# 4. Compute Between-class scatter SB
# -------------------------------
SB = np.zeros((d, d))

for c in classes:
    X_c = X[y == c]
    mean_c = np.mean(X_c, axis=0)
    n_c = X_c.shape[0]
    diff = (mean_c - mean_overall).reshape(d, 1)
    SB += n_c * diff.dot(diff.T)

# -------------------------------
# 5. Solve the generalized eigenvalue problem:
#    SB * w = λ * SW * w
# -------------------------------
eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(SW).dot(SB))

# -------------------------------
# 6. Sort eigenvectors by eigenvalues (descending)
# -------------------------------
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# -------------------------------
# 7. Select top (k-1) eigenvectors for projection
# -------------------------------
W = eigvecs[:, :k-1]   # projection matrix

# -------------------------------
# 8. Project the data
# -------------------------------
X_lda = X.dot(W)

print("Original shape:", X.shape)
print("LDA reduced shape:", X_lda.shape)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2)

axes[0,0].scatter(X_lda[:,0], X_lda[:,1], c=y, cmap="Set1", edgecolors="k")
axes[0,1].scatter(data["feature1"], data["feature2"], c=y, cmap="Set1", edgecolors="k")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA From Scratch (Wine Dataset)")
plt.show()
