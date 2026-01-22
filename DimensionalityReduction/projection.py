import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [2, 3],
    [4, 1],
    [5, 6],
    [-10,4],
    [-2,3]
])

P = np.array([[0, 0], [0, 1]])

projected = X @ P.T

print(projected)

# projected = np.zeros_like(X)   # same shape
projected[:, 1] = X[:, 1]      # keep y-values only

plt.scatter(X[:,0], X[:,1], label="original")

plt.scatter(projected[:,0], projected[:,1], label="projected", color="red")

for i in range(len(X)):
    plt.plot([X[i,0], projected[i,0]], [X[i,1], projected[i,1]], 'k--')

plt.axvline(0, color='black')  # y-axis
plt.legend()
plt.show()