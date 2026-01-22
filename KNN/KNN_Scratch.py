import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy  as np
from collections import  Counter 

def euclidean_dist(x1,x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def  fit(self, X, y):
        self.X_train = X
        self.y_train = y.ravel() 
    
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x):
        distances = [euclidean_dist(x,x_train) for x_train in self.X_train] 
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common()
        
        return most_common[0][0]
    
def main():

    cmap = ListedColormap(['#FF0000','#00FF00',"#5D5DD7"])
    cmap2 = ListedColormap(["#6D0909","#217B21","#101056"])
    cmap3 = ListedColormap(["#E4C425","#E4C425","#E4C425"])

    c1 = [0, 0]
    c2 = [2.2, 2.3]
    c3 = [-2.5, -2.5]

    # number of points per cluster
    n = 200

    # generate clusters with Gaussian noise
    cluster1 = np.random.randn(n, 2) + c1
    cluster2 = np.random.randn(n, 2) + c2
    cluster3 = np.random.randn(n, 2) + c3

    target1 = np.full((n,1), 0) 
    target2 = np.full((n,1), 1) 
    target3 = np.full((n,1), 2) 

    # combine into one dataset
    X = np.vstack([cluster1, cluster2, cluster3])
    y = np.vstack([target1, target2, target3])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    fig, axes = plt.subplots(2, 2)   # 2 rows, 2 columns
    axes[0, 0].scatter(X_train[:,0],X_train[:,1], c=y_train, cmap=cmap, edgecolor='k', s=20)
    axes[0, 0].scatter(X_test[:,0],X_test[:,1], c=y_test, cmap=cmap2, edgecolor='k', s=20)



    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    
    result = predictions == y_test.flatten()
    print(result)
    indices = np.where(result == False)[0]
    print(indices)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)
    
    axes[0, 0].scatter(X_test[indices,0],X_test[indices,1], c=y_test[indices], cmap=cmap3, edgecolor='k', s=20)

    plt.show()

if __name__ == "__main__":
    main()
