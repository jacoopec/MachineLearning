import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create a toy dataset
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train an SVM classifier
clf = SVC(kernel='linear')   # try 'rbf' or 'poly' for non-linear boundaries
clf.fit(X_train, y_train)

# 4. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 5. Visualize the decision boundary
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=30, edgecolors='k')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 50)
yy = np.linspace(ylim[0], ylim[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot margins and boundary
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
           alpha=0.7, linestyles=['--', '-', '--'])

plt.title("SVM Decision Boundary")
plt.show()
