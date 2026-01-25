import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random

# Define mappings
feature_dictionary = {0:"feature1",1:"feature2",2:"feature2",3:"feature3",4:"feature4",5:"feature5"}


data = {
    "feature1": [1, 2, 3, 8, 9, 10, 2, 3, 7, 8],
    "feature2": [5, 4, 6, 7, 8, 9, 3, 2, 6, 5],
    "feature3": [0, 1, 2, 3, 3, 2, 1, 0, 2, 3],
    "feature4": [10, 11, 12, 20, 21, 22, 13, 14, 19, 18],
    "feature5": [100, 105, 102, 200, 202, 199, 110, 111, 180, 178],
    "target":   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
}



def main():
    global data
    feats = ["feature1","feature2","feature3","feature4","feature5"]
    n_feats = random.randint(0, len(feats)) 
    new_data = np.array([])
    
    target = data["target"]
    
    print(n_feats)
    
    for i in range(n_feats):
        new_data = np.column_stack((data[feats[n_feats]]))
    
    print(new_data)
    
    samples = np.zeros((0, 2))
    samp_to_take = 4
    for i in range(samp_to_take):
        x = random.randint(0, len(target)) 
        samples = np.vstack((samples, data[x]))
    print(samples)
    print(data[x] )
    
    
if __name__ == "__main__":
    main()