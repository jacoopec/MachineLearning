import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Define mappings
feature_dictionary = {0:"Weather",1:"Temperature"}
weather_map = {"Sunny": 0, "Rainy": 1, "Windy": 2}
weather_map_inv = {0: "Sunny", 1: "Rainy", 2: "Windy"}
temperature_map = {"Cold": 0, "Normal": 1, "Hot": 2}
temperature_map_inv = {0: "Cold", 1: "Normal", 2: "Hot"}
play_map = {"No": 0, "Yes": 1}


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5,  nFeatures=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.nFeatures = nFeatures
        self.root = None
        self.nodes = []
    
    def fit(self, X, y):
        self.nFeatures = X.shape[1]
        self.root = self._grow_tree(X,y)
        
    def printTree(self):
        for key in self.nodes:
            print(key)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, nfeats = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels==1 or n_samples  < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        feat_idxs = np.random.choice(nfeats, self.nFeatures, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        
        left  = self._grow_tree(X[left_idxs, :],  y[left_idxs],  depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        f = None
        t = None
        
        if(feature_dictionary[best_feature]=="Weather"):
            f  = feature_dictionary[best_feature]=="Weather"
            t = temperature_map_inv[best_threshold]
        else:
            f  = feature_dictionary[best_feature]=="Temperature"
            t = temperature_map_inv[best_threshold]
            
        node = {"best_feature":f,"best_threshold":t}
        self.nodes.append(node)
        return Node(best_feature, best_threshold, left, right)
        
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    
        
        self._print_feature(split_idx, split_threshold)
        return split_idx, split_threshold
    
    def _print_feature(self, split_idx, split_threshold):
        print(feature_dictionary[split_idx])
        if(feature_dictionary[split_idx]=="Weather"):
            print(weather_map_inv[split_threshold])
        else:
            print(temperature_map_inv[split_threshold])
    
    def _information_gain(self, y, X_column, thr):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, thr)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _most_common_label(self,  y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        # print("Traversing tree")
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            self._print_feature(node.feature, node.threshold)
            return self._traverse_tree(x, node.left)
        else:
            self._print_feature(node.feature, node.threshold)
            return self._traverse_tree(x, node.right)
    
    



def main():
    data = [["Sunny", "Sunny", "Sunny", "Windy", "Rainy", "Rainy", "Rainy", "Windy", "Sunny", "Windy", "Sunny"],
        ["Cold",  "Normal", "Hot",   "Hot",   "Cold",  "Hot",  "Normal", "Cold",  "Cold", "Normal", "Hot"],
        ["No",    "Yes",    "Yes",   "Yes",    "No",    "No",   "No",     "No",   "No",    "No",    "Yes"]
    ]
    # Convert to numpy arrays
    weather_num = np.array([weather_map[w] for w in data[0]])
    temperature_num = np.array([temperature_map[t] for t in data[1]])
    play_num = np.array([play_map[p] for p in data[2]])

    data = np.column_stack((weather_num, temperature_num))
    target = play_num

    X, y = data, target

    weather_test = np.array(["Sunny", "Sunny",  "Sunny"])
    temperature_test = np.array(["Cold",  "Normal", "Hot"])
    X_test = np.column_stack((weather_num, temperature_num))

    y_test = np.array(["No",    "Yes",    "Yes"])

    clf = DecisionTree(max_depth = 10)
    clf.fit(X, y)
    predictions = clf.predict(X_test)

    print("tree")
    print(clf.printTree())

if __name__ == "__main__":
    main()