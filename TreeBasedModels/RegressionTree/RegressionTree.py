import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Fits a regression tree to your data
# Input data
X = np.array([0.3, 0.6, 0.65, 0.8, 0.95, 1.2, 1.23, 1.3, 1.45, 1.49,
              1.55, 1.7, 1.73, 1.79, 1.88, 1.98, 2.03, 2.05, 2.1]).reshape(-1, 1)
Y = np.array([0.9, 1.2, 1.15, 1.3, 2.00, 19.3, 20.01, 20.02, 21.02, 22,
              20.7, 14.98, 15.05, 15.9, 15.2, 3.03, 2.69, 2.98, 3.1])

# Train the regression tree
tree = DecisionTreeRegressor(max_depth=0)
tree.fit(X, Y)

# Generate points for smooth curve
X_test = np.linspace(min(X.ravel()), max(X.ravel()), 500).reshape(-1, 1)
Y_pred = tree.predict(X_test)

# Plot original data and tree prediction
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression Tree Prediction')
plt.title('Regression Tree Prediction (max_depth=3)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
