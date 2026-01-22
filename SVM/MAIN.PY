import numpy as np
from svm import LinearSVM 

# Simple linearly separable dataset
X = np.array([
    [2, 3],
    [3, 4],
    [1, 1],
    [-1, -1],
    [-2, -3],
    [-3, -4]
])

y = np.array([1, 1, 1, -1, -1, -1])

svm = LinearSVM(lr=0.01, C=1.0, epochs=1000)
svm.fit(X, y)

predictions = svm.predict(X)
print(predictions)
