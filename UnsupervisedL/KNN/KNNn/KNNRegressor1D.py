import numpy as np
from scipy.spatial import distance

def knn_regressor(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        print(x)
        # Compute distances
        dists = [distance.euclidean(x, x_train) for x_train in X_train]
        print("distances " )
        print( dists)
        # Get indices of k nearest neighbors
        neighbors_idx = np.argsort(dists)[:k]
        print("neighbors_idx " )
        print( X_train[neighbors_idx])
        # Average their target values, it is a linear interpolation
        pred = np.mean([y_train[i] for i in neighbors_idx])
        print(pred)
        predictions.append(pred)
    return np.array(predictions)

# Example usage
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2.1, 2.9, 3.7, 4.2, 5.0])
X_test = np.array([[1.5], [3.5]])

preds = knn_regressor(X_train, y_train, X_test, k=2)
print(preds)