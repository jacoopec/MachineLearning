from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn import datasets
import numpy as np
data = datasets.load_breast_cancer()

data = [["Sunny", "Sunny", "Sunny", "Windy", "Rainy", "Rainy", "Rainy", "Windy", "Sunny", "Windy", "Sunny"],
     ["Cold", "Normal", "Hot", "Hot", "Cold", "Hot", "Normal", "Cold", "Cold", "Normal", "Hot"],
    ["No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes"]
]

# Define mappings
weather_map = {"Sunny": 0, "Rainy": 1, "Windy": 2}
temperature_map = {"Cold": 0, "Normal": 1, "Hot": 2}
play_map = {"No": 0, "Yes": 1}

# Convert to numpy arrays
weather_num = np.array([weather_map[w] for w in data[0]])
temperature_num = np.array([temperature_map[t] for t in data[1]])
play_num = np.array([play_map[p] for p in data[2]])

data = [weather_num,temperature_num,play_num ]

print(data)