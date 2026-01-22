import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Even in 1D, an SVM finds the line (point) that separates the classes with maximum margin.
# The decision function is linear in 1D, so it's just a threshold.
# Maximum margin classfier

# Step 1: Create simple 1D dataset
X = np.array([[1], [2], [3], [6], [7], [8]])  # feature
y = np.array([0, 0, 0, 1, 1, 1])              # labels: 0 or 1

# Step 2: Train a linear SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# Step 3: Predict over a range for visualization
x_range = np.linspace(0, 10, 500).reshape(-1, 1)
y_pred = model.decision_function(x_range)

# Step 4: Plot
plt.figure(figsize=(8, 4))
plt.scatter(X[y == 0], np.zeros_like(X[y == 0]), color='blue', label='Class 0')
plt.scatter(X[y == 1], np.zeros_like(X[y == 1]), color='red', label='Class 1')
plt.plot(x_range, y_pred, color='green', label='Decision Function')
plt.axhline(0, color='black', linestyle='--', label='Decision Boundary')
plt.axhline(1, color='grey', linestyle=':', label='Margin')
plt.axhline(-1, color='grey', linestyle=':')

plt.title('1D SVM Classification')
plt.xlabel('Feature Value')
plt.legend()
plt.grid(True)
plt.show()
