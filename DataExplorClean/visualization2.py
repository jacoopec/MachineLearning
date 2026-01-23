from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

print(data.feature_names)
df = pd.DataFrame(data.data, columns = data.feature_names)

# print(X[:5])
# print(y)
fig, axes = plt.subplots(2, 2)
axes[0,0].scatter(df["sepal length (cm)"], df["sepal width (cm)"])
axes[1,0].scatter(df["sepal length (cm)"], df["sepal width (cm)"])
axes[0,1].scatter(df["petal length (cm)"], df["sepal width (cm)"])
axes[1,1].scatter(df["petal width (cm)"], df["sepal width (cm)"])
plt.tight_layout()
plt.show()

# Class separation
# Correlations
# Feature relationships