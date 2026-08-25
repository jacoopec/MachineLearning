# Regression Metrics in Machine Learning

## Overview

This document describes the most commonly used **evaluation metrics for regression models** in Machine Learning.
Regression tasks involve predicting **continuous numerical values**, and the choice of evaluation metric
has a strong impact on how model performance is interpreted.

---

## Why Regression Metrics Matter

Different regression metrics emphasize different aspects of prediction error.
Some penalize large errors heavily, while others treat all errors equally.

Choosing the right metric depends on:
- The scale of the target variable
- Sensitivity to outliers
- Interpretability requirements
- Business or domain-specific error costs

---

## Common Regression Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and true values.

$$MAE = (1 / n) * Σ |y − ŷ|$$

**Characteristics**
- Easy to interpret
- Robust to outliers

**Use when**
- All errors should be weighted equally

---

### Mean Squared Error (MSE)

Measures the average squared difference between predicted and true values.

$$MSE = (1 / n) * Σ (y − ŷ)²$$

**Characteristics**
- Penalizes large errors more heavily
- Sensitive to outliers

**Use when**
- Large errors are especially undesirable

---

### Root Mean Squared Error (RMSE)

Square root of the Mean Squared Error.

$$RMSE = √MSE$$

**Characteristics**
- Same unit as target variable
- Easier to interpret than MSE

---

### R² Score (Coefficient of Determination)

Measures the proportion of variance in the target variable explained by the model.

$$R² = 1 − (SS_res / SS_tot)$$

**Interpretation**
- 1.0 → perfect fit
- 0.0 → model performs like predicting the mean
- Negative values → model performs worse than the mean

**Limitations**
- Can be misleading for non-linear models

---

### Mean Absolute Percentage Error (MAPE)

Measures prediction error as a percentage.

$$MAPE = (1 / n) * Σ |(y − ŷ) / y| × 100$$

**Limitations**
- Undefined when true values are zero
- Sensitive to small denominators

---

## Metric Selection Guide

| Scenario | Recommended Metrics |
|--------|---------------------|
| Robust to outliers | MAE |
| Penalize large errors | MSE, RMSE |
| Interpretability | MAE, RMSE |
| Model comparison | RMSE, R² |
| Percentage error needed | MAPE |

---

## Key Takeaways

- No single regression metric is universally best
- MAE treats all errors equally, while MSE/RMSE emphasize large errors
- R² measures explained variance, not prediction accuracy
- Metric choice should align with problem goals and data characteristics

---

## References

- Scikit-learn documentation on regression metrics
- An Introduction to Statistical Learning — James et al.
