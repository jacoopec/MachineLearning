import numpy as np
import matplotlib.pyplot as plt

def compute_roc(y_true, y_scores):
    # Sort scores and corresponding true labels (descending)
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Count positives and negatives
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)

    tpr_list = []
    fpr_list = []

    TP = 0
    FP = 0

    # Initial point (threshold above max score)
    tpr_list.append(0.0)
    fpr_list.append(0.0)

    # Sweep threshold from high to low
    for label in y_true_sorted:
        if label == 1:
            TP += 1
        else:
            FP += 1

        TPR = TP / P
        FPR = FP / N

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return np.array(fpr_list), np.array(tpr_list)



def compute_precision_recall(y_true, y_scores):
    # Sort by descending score
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]

    P = np.sum(y_true_sorted == 1)  # total positives

    precision_list = []
    recall_list = []

    TP = 0
    FP = 0

    for label in y_true_sorted:
        if label == 1:
            TP += 1
        else:
            FP += 1

        precision = TP / (TP + FP)
        recall = TP / P

        precision_list.append(precision)
        recall_list.append(recall)

    return np.array(recall_list), np.array(precision_list)



def compute_auc(fpr, tpr):
    return np.trapz(tpr, fpr)






# True labels: 1 = positive, 0 = negative
y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

# Model scores (probabilities or confidence scores)
y_scores = np.array([0.95, 0.90, 0.80, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05])

fpr, tpr = compute_roc(y_true, y_scores)
auc = compute_auc(fpr, tpr)


recall, precision = compute_precision_recall(y_true, y_scores)

print("Recall:", recall)
print("Precision:", precision)



print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", auc)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (from scratch)")
plt.legend()



plt.figure()
plt.plot(recall, precision, marker="o", label="PR curve")

# Baseline = prevalence of positive class
baseline = np.mean(y_true)
plt.hlines(baseline, 0, 1, linestyles="--", label="Baseline")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (from scratch)")
plt.legend()
plt.show()
