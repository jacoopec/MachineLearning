# Classification Metrics in Machine Learning
are a specific category of evaluation metrics designed for classification tasks.



---

## Why Evaluation Metrics ar important 

Imbalanced datasets.
A model may perform well according to one metric and poorly according to another.
Evaluation metrics help quantify model performance beyond simple accuracy and provide insight into error behavior.

---

## Confusion Matrix

Most classification metrics are derived from the confusion matrix:

|                | Predicted Positive | Predicted Negative |
|----------------|-------------------|-------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

---

## Common Classification Metrics

### Accuracy

Measures the proportion of correct predictions.

$$Accuracy = (TP + TN) / (TP + TN + FP + FN)$$

- Simple and intuitive
- Misleading for imbalanced datasets

---

### Precision

Measures how many predicted positive samples are actually positive.

$$Precision = TP / (TP + FP)$$

Use when false positives are costly (e.g. spam detection)

---

### Recall (Sensitivity)

Measures how many actual positives are correctly identified.

$$Recall = TP / (TP + FN)$$

Use when false negatives are costly (e.g. medical diagnosis)

---

### F1-Score

The harmonic mean of precision and recall.

$$F1 = 2 * (Precision * Recall) / (Precision + Recall)$$

Use when data is imbalanced or both precision and recall are important

---

### ROC Curve and AUC

The ROC curve plots the True Positive Rate against the False Positive Rate at different thresholds.
The Area Under the Curve (AUC) summarizes the ROC curve into a single value.

Interpretation:
- AUC = 1.0 : Perfect classifier
- AUC = 0.5 : Random guessing

---

### Precision–Recall Curve

Plots precision versus recall.
More informative than ROC curves when dealing with highly imbalanced datasets.

---

## Metric Selection Guide

| Scenario | Recommended Metrics |
|--------|---------------------|
| Balanced dataset | Accuracy, ROC–AUC |
| Imbalanced dataset | F1-score, PR–AUC |
| Medical / safety-critical | Recall |
| Spam / fraud detection | Precision |
| Model comparison | ROC–AUC |

---

## Key Takeaways

- There is no single best metric for all problems
- Accuracy alone is often insufficient
- Metric choice should reflect the real-world cost of errors
- ROC and Precision–Recall curves provide deeper insight than scalar metrics
