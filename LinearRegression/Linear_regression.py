import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression:
    
    def __init__(self,  lr=0.001, n_iters = 100):
        self.lr = lr 
        self.n_iters  = n_iters
        self.weihts = None
        self.bias = None
        self.weightsList = []
        self.biasesList =  []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)
            

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
            self.weightsList.append(self.weights)
            self.biasesList.append(self.bias)
            
    def predict(self,  X):
        y_pred =  np.dot(X,  self.weights) + self.bias
        return y_pred
    
    


def main():
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=30, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234)

    def mse(y_test, predictions):
        return np.mean((y_test-predictions)**2)
    
    reg = LinearRegression(lr=0.1,n_iters =1000)
    reg.fit(X_train,y_train)
    predictions = reg.predict(X_test)
    iters =  1000
    x = np.linspace(0, iters, iters)
    mse = mse(y_test, predictions)
    print(mse)

    y_pred_line = reg.predict(X)
    cmap = plt.get_cmap('viridis')
    
    fig, axes = plt.subplots(2, 2)   # 2 rows, 2 columns
    axes[0, 0].scatter(X_train, y_train, color="green", s=10)
    axes[0, 0].scatter(X_test, y_test, color="red", s=10)
    axes[0, 0].plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    axes[1, 1].scatter(x, reg.weightsList, color="green", s=10)
    axes[1, 0].scatter(x, reg.biasesList, color="green", s=10)
    
    plt.show()

if __name__ == "__main__":
    main()

