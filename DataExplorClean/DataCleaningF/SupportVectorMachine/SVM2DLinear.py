import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# This is a support vector machine in 2D with linear kernel

# Step 1: Create synthetic 2D dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 8],
    [8, 8]
])
y = np.array([0, 0, 0, 1, 1, 1])  # Labels

# Step 2: Train a linear SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# Step 3: Plotting the decision boundary and margins
w = model.coef_[0]
b = model.intercept_[0]

# Create grid to plot
xx, yy = np.meshgrid(np.linspace(0, 10, 500), np.linspace(0, 10, 500))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Class 1')

# Plot decision boundary and margins
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['gray', 'black', 'gray'], linestyles=['--', '-', '--'])

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.title("2D SVM with Linear Kernel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
