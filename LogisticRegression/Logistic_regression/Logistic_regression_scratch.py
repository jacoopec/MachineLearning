import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.sigmoid = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            self.sigmoid = sigmoid(linear_pred)

            X = X.reshape(-1, 1)   # (n_samples, 1)
            y = y.reshape(-1, 1)   # (n_samples, 1)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        X = X.reshape(-1, 1)   # (n_samples, 1)
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        y_pred = y_pred.reshape(-1, 1)   # (n_samples, 1)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    



# bc = datasets.load_breast_cancer()
# X, y = bc.data[:,:1], bc.target

X = np.array([[
    0.1, 0.3, 0.5, 0.6, 0.8, 1.2, 1.8, 2.4,
    3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0,
    6.4, 6.8, 7.3, 7.7, 8.1, 8.6, 9.0, 9.6
]])

y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1
])

X_train = X_test = X
y_train = y_test = y




clf = LogisticRegression(lr = 0.1,n_iters=3)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(clf.sigmoid.shape)

plt.scatter(X, y)
X = X.reshape(-1, 1)   # (n_samples, 1)
plt.plot(X, clf.sigmoid )
plt.xlabel("X")
plt.ylabel("y")
plt.title("Data points")
plt.yticks([0, 1])
plt.show()

