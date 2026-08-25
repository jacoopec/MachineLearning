from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

# 1. Example binary classification dataset
X, y = make_classification(
    n_samples=8,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

print(X.shape)
print(y.shape)
values, counts = np.unique(y, return_counts=True)
print(values)   # [1 2 3 4]
print(counts)   # [1 2 3 1]


# 2. Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

# 3. Stratified K-Fold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 4. Cross-validated predicted probabilities
#    cross_val_predict returns out-of-fold predictions
y_proba = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)
print(y_proba)


# Probability for positive class
y_score = y_proba[:, 1]

# 5. Convert probabilities to class labels with threshold 0.5
y_pred = (y_score >= 0.5).astype(int)

# 6. Evaluate
auc = roc_auc_score(y, y_score)

print("Predicted probabilities shape:", y_proba.shape)
print("First 5 probabilities:\n", y_proba[:5])
print("ROC AUC:", round(auc, 4))
print("\nClassification report:\n")
print(classification_report(y, y_pred))