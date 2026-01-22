import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load your .ods file (replace with your actual path)
df = pd.read_excel("data.xls")

# Drop the first column (ID)
df = df.iloc[:, 1:]

# Convert 'y'/'n' to 1/0
df = df.applymap(lambda x: 1 if x == 'y' else 0)

# Split features and label
X = df.iloc[:, 0:-1]  # All columns except last
y = df.iloc[:, -1]   # Last column is the label


# Logistic Regression model
model = RandomForestClassifier()
model.fit(X, y)

# Predict
y_pred = model.predict(X)
print(y_pred)

y_test = y.to_numpy()
print(y_test)
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
