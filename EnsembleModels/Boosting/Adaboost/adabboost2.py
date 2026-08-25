import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

index = 0

# data = {
#     "Weather":     ["Sunny", "Sunny",  "Sunny", "Windy", "Rainy",  "Rainy",  "Rainy", "Windy",  "Sunny",   "Windy",    "Sunny"],
#     "Temperature": ["Cold",   "Normal", "Hot",   "Hot",   "Cold",   "Hot",   "Normal", "Cold",    "Cold",   "Normal",   "Hot"],
#     "Play?":         ["No",   "Yes",    "Yes",    "Yes",   "No",    "No",     "No",     "No",     "No",      "No",      "Yes"],
# }

data =  {
    "Sunny":["Yes",   "No", "No", "No",  "Yes", "Yes","No",  "No","No"],
    "Windy":["Yes",  "Yes", "Yes","No",  "No" , "Yes","No",  "Yes","No"],
    "Temp":["High", "High","Low","High","Low", "Low","Low", "High","High"],
    "Rainy":["No",   "No",  "Yes","Yes", "No" , "No", "Yes", "Yes","Yes"],
    "Play?":["Yes",  "Yes", "No", "No",  "Yes", "No", "No",  "No","Yes"]
}

le = LabelEncoder()
# "NumberPlayers": [3, 2, 7, 4, 3, 2, 6, 9, 4],
splits = {}

df = pd.DataFrame(data)

df["Sunny"]     = le.fit_transform(df["Sunny"])
df["Windy"] = le.fit_transform(df["Windy"])
df["Temp"] = le.fit_transform(df["Temp"])
df["Rainy"] = le.fit_transform(df["Rainy"])
df["Play?"]       = le.fit_transform(df["Play?"])



def gini(labels):
    """Compute gini impurity of a group."""
    if len(labels) == 0:
        return 0
    p = np.mean(labels)
    return 1 - (p**2 + (1-p)**2)


def try_splits(feature):
    global index
    values = sorted(df[feature].unique())
    thresholds = values[:-1]  # possible splits
    print(thresholds)
    print(feature)

    for t in thresholds:
        left = df[df[feature] <= t]["Play?"]
        print(left)
        right = df[df[feature] > t]["Play?"]
        print(right)

        # weighted Gini
        g_left = gini(left)
        g_right = gini(right)
        weighted = (len(left)*g_left + len(right)*g_right) / len(df)

        # print(f"  Split: {feature} <= {t}  --> Gini = {weighted:.3f}")
        splits[index] = {"feature": feature, "gini":weighted, "t" : t}
        index = index + 1
    index = index + 1
    
if __name__ == "__main__":
    
    df2 = df.drop("Play?", axis=1)
    for col in df2.columns:
        try_splits(col)
    best = min(splits.items(), key=lambda x: x[1]["gini"])
    
    
    print(best[1])
    print(df)