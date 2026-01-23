import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# The RBF kernel can draw curved decision boundaries that adapt to complex patterns in the data.

#Plot in 2D or 3D:
visualization2D = False

# Step 1: Generate non-linearly separable 2D data
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)

# Step 2: Train an SVM with RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='auto')  # gamma='auto' or 'scale'
model.fit(X, y)

# Step 3: Create grid to plot decision function
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 500), np.linspace(-1.5, 1.5, 500))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Step 4: Plot
if(visualization2D ):
    plt.figure(figsize=(8, 6))

    # Plot data
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['gray', 'black', 'gray'], linestyles=['--', '-', '--'])

    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.title("2D SVM with RBF Kernel (Nonlinear Boundary)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


else:
    
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the decision function surface
    ax.plot_surface(xx, yy, zz, cmap='coolwarm', alpha=0.6)

    # Plot the decision boundary plane (z=0)
    ax.contour3D(xx, yy, zz, levels=[0], colors='black', linewidths=2)

    # Plot training data in 3D (z = decision function at each point)
    z_vals = model.decision_function(X)
    ax.scatter(X[:, 0], X[:, 1], z_vals, c=y, cmap='bwr', edgecolor='k', s=50)

    # Labels
    ax.set_title("3D SVM Decision Surface with Margin Plane")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Decision Function Value")

    plt.tight_layout()
    plt.show()