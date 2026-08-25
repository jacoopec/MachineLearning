import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.array([[1, 2, 3], [4, 5, 6]])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X[:, 0], X[:, 1],X[:, 0])

plt.show()

print(X[:, 0])
print(X[0, :])