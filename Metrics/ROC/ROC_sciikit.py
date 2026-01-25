from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_score = [0.92, 0.30, 0.85, 0.60, 0.40, 0.10, 0.70, 0.20]
y_true = [1, 0, 1, 1, 0, 0, 1, 0]

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()