# Logistic Regression: Log-Likelihood Explanation

## What are we optimizing?

In logistic regression, we optimize the **log-likelihood of the observed class labels** given the input data and the model parameters.

Suppose we have a training dataset:

```text
D = {(x1, y1), (x2, y2), ..., (xn, yn)}
```

where each label is binary:

```text
yi ∈ {0, 1}
```

The logistic regression model estimates the probability that an input belongs to class `1`:

```text
P(yi = 1 | xi; w, b) = ŷi = σ(wᵀxi + b)
```

where:

```text
σ(z) = 1 / (1 + e^(-z))
```

The probability of class `0` is therefore:

```text
P(yi = 0 | xi; w, b) = 1 - ŷi
```

## Probability of one observed label

For one training example, the probability of observing the true label `yi` is:

```text
P(yi | xi; w, b) = ŷi^yi (1 - ŷi)^(1 - yi)
```

This compact formula handles both possible cases.

If the true label is:

```text
yi = 1
```

then:

```text
P(yi | xi; w, b) = ŷi
```

If the true label is:

```text
yi = 0
```

then:

```text
P(yi | xi; w, b) = 1 - ŷi
```

So the model gives high probability to the data when:

- `ŷi` is close to `1` for examples with label `1`
- `ŷi` is close to `0` for examples with label `0`

## Likelihood of the full dataset

Assuming the training examples are independent, the likelihood of the whole dataset is the product of the probabilities of all observed labels:

```text
L(w, b) = ∏ᵢ P(yi | xi; w, b)
```

More explicitly:

```text
L(w, b) = ∏ᵢ ŷi^yi (1 - ŷi)^(1 - yi)
```

The parameters `w` and `b` are chosen to make this likelihood as large as possible.

In simple words:

> We choose the parameters that make the observed training labels as probable as possible.

## Log-likelihood

Products of many probabilities can become very small, so we usually take the logarithm of the likelihood.

The log-likelihood is:

```text
ℓ(w, b) = Σᵢ [ yi log(ŷi) + (1 - yi) log(1 - ŷi) ]
```

Instead of maximizing the likelihood, we maximize the log-likelihood.

This is mathematically equivalent because the logarithm is a monotonic function: if one likelihood is larger than another, its logarithm is also larger.

## Negative log-likelihood and cross-entropy

In machine learning libraries, we usually minimize a loss function instead of maximizing an objective.

So instead of maximizing the log-likelihood:

```text
ℓ(w, b)
```

we minimize the negative log-likelihood:

```text
-ℓ(w, b)
```

This gives:

```text
Loss(w, b) = - Σᵢ [ yi log(ŷi) + (1 - yi) log(1 - ŷi) ]
```

This loss is also called:

- **binary cross-entropy**
- **logistic loss**
- **negative log-likelihood**

## Intuition

For each training example:

- If the true label is `1`, the model is rewarded when `ŷi` is close to `1`
- If the true label is `0`, the model is rewarded when `ŷi` is close to `0`
- If the model assigns low probability to the correct label, the loss becomes large

So logistic regression learns parameters by making the correct labels as likely as possible.

## Summary

Logistic regression estimates its parameters by optimizing the probability of the observed class labels.

The main idea is:

```text
Find w and b that maximize P(y | X; w, b)
```

or equivalently:

```text
Find w and b that minimize binary cross-entropy loss
```

Therefore, logistic regression is trained by maximizing the log-likelihood of the observed labels under the model.
