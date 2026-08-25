# Clustering Metrics in Machine Learning

## Overview

This document introduces the most common evaluation metrics used for clustering algorithms in Machine Learning.
Clustering is typically an unsupervised learning task, meaning that ground truth labels are often not available.
As a result, clustering metrics are divided into internal and external metrics.

---

## Why Clustering Metrics Matter

Clustering algorithms will always produce clusters, but those clusters are not always meaningful.
Evaluation metrics help assess how well the clustering captures the underlying structure of the data.

They are used to evaluate:
- Cluster compactness
- Cluster separation
- Agreement with known labels (when available)

---

## Types of Clustering Metrics

### 1. Internal Metrics (No Ground Truth Required)

Internal metrics evaluate clustering quality using only the data and the predicted cluster labels.

---

### Silhouette Score

Measures how similar a data point is to its own cluster compared to other clusters.

Range:
- -1 to 1

Interpretation:
- Values close to 1 indicate well-separated clusters
- Values around 0 indicate overlapping clusters
- Negative values suggest incorrect clustering

---

### Davies–Bouldin Index

Measures the average similarity between each cluster and its most similar cluster.

Interpretation:
- Lower values indicate better clustering performance

Advantages:
- Computationally efficient

Limitations:
- Sensitive to noise and cluster shape

---

### Calinski–Harabasz Index

Measures the ratio of between-cluster dispersion to within-cluster dispersion.

Interpretation:
- Higher values indicate better-defined clusters

---

## 2. External Metrics (Ground Truth Required)

External metrics compare clustering assignments with known class labels.

---

### Adjusted Rand Index (ARI)

Measures similarity between true labels and cluster assignments, adjusted for chance.

Range:
- -1 to 1

Interpretation:
- 1 indicates perfect agreement
- 0 indicates random labeling

---

### Normalized Mutual Information (NMI)

Measures the mutual dependence between the true labels and the cluster assignments.

Range:
- 0 to 1

Interpretation:
- Higher values indicate stronger agreement

---

### Homogeneity, Completeness, and V-Measure

- Homogeneity: each cluster contains only members of a single class
- Completeness: all members of a class are assigned to the same cluster
- V-measure: harmonic mean of homogeneity and completeness

---

## Metric Selection Guide

| Scenario | Recommended Metrics |
|--------|---------------------|
| No ground truth labels | Silhouette, Davies–Bouldin |
| Choosing number of clusters | Silhouette, Calinski–Harabasz |
| Ground truth available | ARI, NMI |
| Large datasets | Davies–Bouldin |

---

## Key Takeaways

- Clustering evaluation is inherently challenging
- Internal and external metrics serve different purposes
- No single metric fully captures clustering quality
- Metric choice should reflect data characteristics and goals

---

## References

- Scikit-learn documentation on clustering metrics
- An Introduction to Statistical Learning — James et al.
