
from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# data = datasets.load_breast_cancer()
# X = data.data
# y = data.target



# df =  pd.read_fwf("beachvolley_data.txt")
# le = LabelEncoder()
# df['Weather'] = le.fit_transform(df['Weather'])
# df['Temperature'] = le.fit_transform(df['Temperature'])
# df['Play?'] = le.fit_transform(df['Play?'])


# X =  df.iloc[:, :3].to_numpy()
# y = df.iloc[:, -1].to_numpy()




# def accuracy(y_true, y_pred):
#     accuracy = np.sum(y_true == y_pred) / len(y_true)
#     return accuracy


X_train = [[0,3,6,4,3],[0,7,2,9,4],[2,8,6,3,9],[8,3,6,5,0],[5,6,6,3,7]]
y_train = [1,0,1,0,1]
X_test = [[0,3,6,4,3],[0,3,6,4,3],[0,3,6,4,3],[0,3,6,4,3],[0,3,6,4,3]]

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


# acc =  accuracy(y_test, predictions)
# print(acc)