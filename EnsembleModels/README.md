# Ensemble Methods in Machine Learning

## Overview

Ensemble methods are powerful techniques in machine learning that combine multiple models (often called "base learners" or "weak learners") to improve overall performance. By aggregating predictions from several models, ensembles can reduce variance, bias, or both, leading to more accurate and robust results compared to individual models.

The majority of the  K classifiers in the ensambble will be correct on many examples where any individual  classifier makes errors.

Senza  ensamble l'errore medio di n  classificatori su un dataset è dato dalla media della somma degli eerrori. Facendo  l'emmsanble, l'errore su un certo dataset diminuisce perchè faccio la media di n errori quindi, un errore pesa 1/n.
usando l'ensamble l'errore pesa 1/n volte meno rispetto all'errore di ogni classificatore 


Common ensemble methods include:
- **Bagging** (Bootstrap Aggregating)
- **Boosting**
- **Stacking**
- **Random Forests** (a specific type of bagging)
- **Gradient Boosting Machines** (a type of boosting)

These methods are widely used in classification, regression, and other tasks, often winning Kaggle competitions and real-world applications.

## Why Use Ensembles?
- **Improved Accuracy**: Combining models can correct individual errors.
- **Reduced Overfitting**: Especially with methods like bagging.
- **Robustness**: Handles noisy data and outliers better.
- **Interpretability**: Some ensembles (e.g., Random Forests) provide feature importance insights.

## Key Ensemble Methods

### Bagging (Bootstrap Aggregating)
Bagging involves training multiple models on different subsets of the data (created via bootstrapping, i.e., sampling with replacement) and then averaging their predictions (for regression) or taking a majority vote (for classification).

- **How it Works**: Each model is trained independently on a random subset. The final prediction is the aggregate.
- **Benefits**: Reduces variance, making it great for unstable models like decision trees.
- **Example**: Random Forest is bagging applied to decision trees with random feature selection.

**Analogy**:
Bagging: pick learners randomly.  
For example in a test, we could ask help to some friends. They don't have to be genious, but the catch is to gather the answers that each one of them would give and understand which answer is the most likely to be correct, based on the answers they gave.

### Boosting
Boosting builds models sequentially, where each new model focuses on correcting the errors of the previous ones. It assigns higher weights to misclassified samples, turning weak learners into a strong ensemble.

- **How it Works**: Start with a base model, then iteratively add models that emphasize hard-to-predict instances. Predictions are combined via weighted voting.
- **Benefits**: Reduces bias and variance, often achieving high accuracy.
- **Examples**: AdaBoost, Gradient Boosting (e.g., XGBoost, LightGBM).

**Analogy**:
Boosting: picking learners intelligently.  
We exploit the strenght of each friend. So, each friend will compensate for some weakness some other friend.

### Stacking
Stacking (or stacked generalization) trains multiple base models and then uses a meta-model (e.g., logistic regression) to combine their predictions.

- **How it Works**: Base models' outputs become inputs for the meta-learner.
- **Benefits**: Can outperform bagging/boosting by learning optimal combinations.
- **Drawbacks**: More complex and computationally intensive.

## Implementation Tips
- **Libraries**: Use scikit-learn for basic ensembles (e.g., `RandomForestClassifier`, `GradientBoostingClassifier`). For advanced boosting, try XGBoost or LightGBM.
- **Hyperparameters**: Tune `n_estimators` (number of models), `learning_rate` (for boosting), and `max_depth` (for trees).
- **Cross-Validation**: Always use it to avoid overfitting.
- **When to Use**: Ensembles shine on tabular data; for deep learning, consider neural network ensembles.


Gli errori che fanno i classificatori sono indipendenti tra loro.
La maggior parte dei classificatori nell'ensemble sarà corretta sugli esempi dove classificatori singoli fanno errori.
In questo caso si sbaglia a classificare se più della metà dei classificatori fa una predizione sbagliata.
The error follows the cumulative binomial distribution and the  accuravy probability of the ensamble will be greater than that of the single classifiers.

