import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
data = {
    'Age': [
        22, 25, 27, 30, 33, 35, 37, 40, 43, 45,
        47, 49, 50, 52, 54, 56, 58, 60, 62, 64,
        66, 68, 70, 72, 74, 76, 78, 80, 82, 85
    ],
    'Salary': [
        15000, 18000, 20000, 23000, 25000, 28000, 30000, 33000, 36000, 40000,
        45000, 47000, 49000, 52000, 54000, 57000, 59000, 62000, 65000, 70000,
        72000, 75000, 78000, 80000, 82000, 85000, 87000, 89000, 90000, 92000
    ],
    'Purchased': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}


df = pd.DataFrame(data)

# Features and label
X = df[['Age', 'Salary']]
y = df['Purchased']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 🖼️ 6. Optional: Plotting decision boundary (for 2D only)
