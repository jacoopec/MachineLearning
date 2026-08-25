# Machine Learning Experiments


## Supervised Learning
Supervised learning models are models that map inputs to outputs, and attempt to extrapolate patterns learned in past data on unseen data. Supervised learning models can be either regression models, where we try to predict a continuous variable, like stock prices—or classification models, where we try to predict a binary or multi-class variable, like whether a customer will churn or not. In the section below, we'll explain two popular types of supervised learning models: linear models, and tree-based models.


## Unsupervised Learning
Unsupervised learning is about discovering general patterns in data. The most popular example is clustering or segmenting customers and users. This type of segmentation is generalizable and can be applied broadly, such as to documents, companies, and genes. Unsupervised learning consists of clustering models, that learn how to group similar data points together, or association algorithms, that group different data points based on pre-defined rules. 



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


---

## Disclaimer

This folder is **experimental by design**:
- code may be incomplete or exploratory
- approaches may change as understanding improves
- not intended to be used as a production-ready machine learning library
