import numpy as np
# from sklearn import datasets
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
dim = 100
data = np.random.randint(100, size=(dim, 5))
labels = np.random.randint(2,size=(dim))

# print(data)
clf = LogisticRegression()
clf.fit(data,labels)
y_pred = clf.predict(data)

preds = [y_pred[i] == labels[i] for i in y_pred]

print(preds)

indexes = [i for i, value in enumerate(preds) if not value]
print(indexes)


# def accuracy(y_pred, y_test):
#     return np.sum(y_pred==y_test)/len(y_test)

# acc = accuracy(y_pred, labels)
# print(acc)

dataset =  np.array([])

weights =  np.ones([dim])

for j in range(preds):
    dataset.append()
    if preds 
