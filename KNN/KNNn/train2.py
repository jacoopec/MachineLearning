import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
from sklearn.datasets import make_blobs

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])



# Generate sample data
X, y = make_blobs(n_samples=100, centers=3, cluster_std=2.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X_train[:,0],X_train[:,1], c=y_train, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

for p in predictions:
    print(p[0])

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)