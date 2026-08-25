import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example multi-class data
y_true = np.array([0, 1, 2, 2, 0, 1, 0, 2, 1])
y_pred = np.array([0, 2, 2, 2, 0, 1, 1, 2, 1])

# Get all class labels
classes = np.unique(np.concatenate((y_true, y_pred)))
n_classes = len(classes)

# Initialize confusion matrix
conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

# Build confusion matrix manually
for t, p in zip(y_true, y_pred):
    conf_matrix[t][p] += 1

# Print confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Compute precision, recall, and F1-score per class
precision = []
recall = []
f1_score = []

for i in range(n_classes):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

# Display per-class metrics
for i, cls in enumerate(classes):
    print(f"\nClass {cls}:")
    print(f"  Precision: {precision[i]:.2f}")
    print(f"  Recall:    {recall[i]:.2f}")
    print(f"  F1-score:  {f1_score[i]:.2f}")

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
