import pandas as pd
import numpy as np

def make_categorical_stump(df, feature, category, target):
    """
    Creates a decision stump for categorical features.
    
    df        : pandas DataFrame
    feature   : categorical column to split on
    category  : value used for the split
    target    : target column (class label)
    
    Returns a dictionary with left and right leaf class distributions.
    """
    
    # Split: matching category vs not matching
    left_leaf = df[df[feature] == category]
    right_leaf = df[df[feature] != category]
    
    # Class distributions
    left_dist = left_leaf[target].value_counts().to_dict()
    right_dist = right_leaf[target].value_counts().to_dict()
    
    
    gini_index = 1 - ((df[feature] == category).sum() / df.shape[0])**2 - ((df[feature] == category).sum() / df.shape[0])**2
    print(gini_index)
    
    return {
        "split_feature": feature,
        "category": category,
        "left_leaf": left_dist,
        "right_leaf": right_dist
    }


def make_stump(df, feature, threshold, target):
    """
    Creates a decision stump: a one-level decision tree.
    
    df        : pandas DataFrame
    feature   : column name to split on
    threshold : numeric or categorical threshold
    target    : target column (class label)
    
    Returns a dictionary with the left and right leaf class distributions.
    """
    
    # Split the data
    left_leaf = df[df[feature] <= threshold]
    right_leaf = df[df[feature] > threshold]
    
    # Class distributions
    left_dist = left_leaf[target].value_counts().to_dict()
    right_dist = right_leaf[target].value_counts().to_dict()
    
    return {
        "split_feature": feature,
        "threshold": threshold,
        "left_leaf": left_dist,
        "right_leaf": right_dist
    }
    
def compute_gini(df, target):
    values = df[df.columns[0]].unique()
    
    giniIndexes = np.zeros_like(values)

    index = 0
    
    # print(values)

    for val in values:
        # print(val)
        subsetYes = df[df[df.columns[0]] == val]  
        subsetNo = df[df[df.columns[0]] != val]  
        gini_indexYes = 1 - ((subsetYes[target] == "Yes").sum() / subsetYes.shape[0])**2 - ((subsetYes[target] == "No").sum() / subsetYes.shape[0])**2
        gini_indexNo = 1 - ((subsetNo[target] == "Yes").sum() / subsetNo.shape[0])**2 - ((subsetNo[target] == "No").sum() / subsetNo.shape[0])**2
        gini_impurity =  gini_indexYes*((subsetYes[target] == "Yes").sum()  + (subsetYes[target] == "No").sum()) / subsetYes.shape[0] + gini_indexNo*((subsetNo[target] == "Yes").sum()  + (subsetNo[target] == "No").sum()) / subsetNo.shape[0] 
        giniIndexes[index] = gini_impurity
        index = index + 1
        # print(giniIndexes)
    
    best_split_feature = values[np.where(giniIndexes == max(giniIndexes))[0][0]]
    print(f"The best split for the feature {df.columns[0]} is {max(giniIndexes)} for {best_split_feature}")
    
    return {
        "best_split_feature":best_split_feature,
        "max(giniIndexes)":max(giniIndexes)
    }

df = pd.read_csv(
    "beachvolley_data.txt",
    sep=r"\s+",
    engine="python"
)

arr = df.to_numpy()


