import numpy as np
from sklearn.preprocessing import OrdinalEncoder

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean   = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var    = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes,               dtype = np.float64)

        for idx, c in enumerate(self._classes):
            print(X)
            print(y == c)
            X_c = X[y == c]
            print(f"for {c} , {X_c.size} of {X.size}, {y}, {c}, {y == c}" )
            # print(X_c)
            self._mean[idx, :] = X_c.mean(axis = 0)
            self._var[idx, :] = X_c.var(axis = 0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            print("mean")
            print(self._mean)
            print("var")
            print(self._var)
            print("priors")
            print(self._priors)
            

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
    import matplotlib.pyplot as plt
    

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # X, y = datasets.make_classification(
    #     n_samples=50,
    #     n_features=2,
    #     n_informative=2,
    #     n_redundant=0,
    #     n_repeated=0,
    #     n_classes=2,
    #     random_state=123
    # )
    
    # X = np.array([[1,0,1,1,1,0,0,0,0,1],
    #               [0,0,1,1,1,0,0,0,0,1]])
    # y = np.array([[1,1,1,1,1,0,0,0,0,1]])
    

    X = np.array([
        ["Sunny",    "Hot",  "High",   "False"],
        ["Sunny",    "Hot",  "High",   "True",],
        ["Overcast", "Hot",  "High",   "False"],
        ["Rain",     "Mild", "High",   "False"],
        ["Rain",     "Cool", "Normal", "False"],
        ["Rain",     "Cool", "Normal", "True",],
        ["Overcast", "Cool", "Normal", "True",],
        ["Sunny",    "Mild", "High",   "False"],
        ["Sunny",    "Cool", "Normal", "False"],
        ["Rain",     "Mild", "Normal", "False"],
        ["Sunny",    "Mild", "Normal", "True",],
        ["Overcast", "Mild", "High",   "True",],
        ["Overcast", "Hot",  "Normal", "False"],
        ["Rain",     "Mild", "High",   "True",]
    ])
    y = np.array(["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"])
    # y = y.reshape(-1, 1)
    mapping = {"No": 0, "Yes": 1}
    y = np.array([mapping[label] for label in y])
    encoder = OrdinalEncoder()
    X = encoder.fit_transform(X)
    print(X)
    print(y)
    
    
    plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Synthetic Classification Data")
    plt.show()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    # X_train, X_test = X
    # y_train, y_test = y

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)


    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))