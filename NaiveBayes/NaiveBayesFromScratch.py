import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            print(self._mean)
            

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    n_samples = 300  # Total points
    centers = [[-1, -1], [0, 0]]  # Cluster centers for diagonal separation
    cluster_std = 1.0  # Standard deviation for overlap

    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=cluster_std,
        random_state=80  # For reproducibility
    )
    
    print(X.size)
    print(y.size)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print(predictions[50:90])
    print(y_test[50:90])

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))