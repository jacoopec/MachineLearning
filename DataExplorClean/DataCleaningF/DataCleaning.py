from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df =  pd.read_fwf("Datacleaning")

#HANDLE MISSING VALUES
#missing values per column
df.isnull().sum() 
#remove 
df = df.dropna()
#FILL MISSING WITH MEAN 
df['column'] = df['column'].fillna(df['column'].mean())
#CUSTOM VALUE
df['column'] = df['column'].fillna(0)
#Most frequent value 
df['column'] = df['column'].fillna(df['column'].mode()[0])





# ENCODE CATEGORICAL FEATURES
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})





#Convert to Category or Numeric (if needed)
df = pd.DataFrame({'Play?': ['Yes', ' no', 'YES', 'No', 'yes', ' NO', 'yes']})

# Clean labels
df['Play?'] = df['Play?'].str.strip().str.lower().replace({'yes': 'Yes', 'no': 'No'})

# Final categories
print(df['Play?'].value_counts())
