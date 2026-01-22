from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

X = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18]]
y = [0,0,0,0,1,0,1,0,0,1,1,0,1,1,1,1,1,1,1]
thresholds = np.arange(0.0, 1.0, 1/18)
tp = []
fp = []

def confusion_counts(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")

    TP = TN = FP = FN = 0

    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 0:
            TN += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:
            FN += 1

    return TP, TN, FP, FN



model = LogisticRegression()
model.fit(X, y)

probs = model.predict_proba(X)[:, 1]  # probability of class 1

for t in thresholds:
    y_pred_custom = (probs >= t).astype(int)
    TP, TN, FP, FN = confusion_counts(y, y_pred_custom)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    fp.append(FPR)
    tp.append(TPR)

plt.plot(fp, tp)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot")
plt.show()
print(tp)
print(fp)