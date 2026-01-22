# Machine Learning Experiments

This folder contains a collection of **machine learning experiments** focused on understanding how algorithms work, rather than building production-ready systems.

The experiments range from **implementing algorithms from scratch** to **applying models to small or controlled datasets** in order to study their behavior, performance, and evaluation metrics.

---

## Purpose

The main goal of this folder is **learning through experimentation**.

In particular, it is used to:
- gain a deeper understanding of core machine learning algorithms
- study the impact of different **design choices and hyperparameters**
- analyze model performance using multiple **evaluation metrics**
- bridge the gap between theoretical concepts and practical implementation

---

## Types of Experiments

### Algorithm Implementation from Scratch
- Reimplementation of classic machine learning algorithms without relying on high-level libraries
- Focus on:
  - mathematical foundations
  - training and optimization logic
  - model behavior and limitations

Examples include:
- Linear and Logistic Regression
- k-Nearest Neighbors
- Decision Trees
- Simple Neural Networks

---

### Experiments on Small Datasets
- Use of **small, synthetic, or toy datasets**
- Emphasis on interpretability rather than raw performance
- Useful for:
  - visualizing decision boundaries
  - observing overfitting and underfitting
  - understanding bias–variance trade-offs

---

### Configuration and Metric Evaluation
- Running the same model with different:
  - hyperparameters
  - preprocessing techniques
  - model configurations
- Comparing results using metrics such as:
  - Accuracy
  - Precision / Recall
  - F1-score
  - RMSE / MAE

The objective is to understand how configuration changes affect performance.

---

## Typical Workflow

1. Choose an algorithm or concept to explore  
2. Implement it from scratch or apply an existing implementation  
3. Run experiments on small or well-understood datasets  
4. Evaluate results using appropriate metrics  
5. Analyze strengths, weaknesses, and limitations  

---

## Folder Organization (example)

```
.
├── from_scratch/
│   ├── linear_regression/
│   ├── decision_tree/
│   └── neural_networks/
├── small_datasets/
│   ├── synthetic/
│   └── toy_examples/
├── metrics_analysis/
│   ├── hyperparameter_study/
│   └── model_comparison/
└── notebooks/
```

*(The structure may evolve as experiments grow.)*

---

## Notes

- Code clarity and correctness are prioritized over optimization
- Datasets are intentionally small unless otherwise specified
- Results are meant for **analysis and learning**, not benchmarking

---

## Disclaimer

This folder is **experimental by design**:
- code may be incomplete or exploratory
- approaches may change as understanding improves
- not intended to be used as a production-ready machine learning library
