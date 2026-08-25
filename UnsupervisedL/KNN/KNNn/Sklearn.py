from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Generating data
X, y = make_blobs(n_samples=100, centers=3, cluster_std=2.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

for i in range(0, len(X_test)):
    print(X_test[i], y_pred[i], y_test[i])