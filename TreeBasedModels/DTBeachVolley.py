from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv("beachvolley_data_augmented.txt", sep="\s+")

y = data["Play?"]
data = data.drop(columns=["Play?"])


le = LabelEncoder()

for col in data.columns:
    data[col] = le.fit_transform(data[col])


data["Weather"] = le.fit_transform(data["Weather"])
data["Temperature"] = le.fit_transform(data["Temperature"])

print(data["Weather"].shape)
print(data["Temperature"].shape)
subset = data

X_train, X_test, y_train, y_test = train_test_split(
    subset, y, test_size=0.3, random_state=42
)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


text_tree = tree.export_text(model, feature_names=list(subset.columns))
print(text_tree)