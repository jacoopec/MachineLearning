from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)   # converge easily
model.fit(X_train, y_train)

print(data.keys())
print(data.feature_names)

df = pd.DataFrame(data.data, columns=data.feature_names)

# plt.scatter(df['alcohol'], df['malic_acid'])
# plt.scatter(df['alcohol'], df['magnesium'])
plt.scatter(df['color_intensity'], df['magnesium'])
plt.show()

print(df['alcohol'].mean)