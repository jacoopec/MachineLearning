# Logistic Regression 


addestra a maximum likelihood 
occorre massimizzare lo score della classe giusta. 

risolvere il problema  di maximum likelihood nel caso della log reg risolve  un problema di minimizzazione della negative cross entropy.
equivalenza  tra queste due metriche.

la cross entropy non è altro chela likelihood negata di un log reg classificatore

è  una  rete  neurale  senza  strati intermedi  
la log reg è  un classificatore descriminativo 

generativo:
Se il  classificatore ha una  rappresentazione p(x|y)  che mi permette di campionare
che mi permette di campionare x nota la classe.

discriminativo:
si costruiscono uno  score o ddensità  di probabilità p(y|x),  chenon mi  permette di campionare 
x,ma  mi permette di calcolare lo score della classe y dato x.

fornisce la probabilità a posteriori nella log  reg viene calcolata direttamente.
Non si  applica la regola di bayes ma si calcola direttamente P(y|x)

è un  classificatore lineare,  utilizza  una  retta(piano ) per separare le due classi.



is a classification algorithm, not a regression algorithm (despite the name).
It’s used to `predict a binary outcome` (like yes/no, 0/1, spam/not spam) based on one or more input features.
Instead of predicting a raw number like in linear regression, logistic regression predicts a probability that a given input belongs to class 1.

$$z = w * x + b $$
$$y = 1 / 1 + e^-z$$


𝑦^ is the predicted probability (between 0 and 1)

The final class is predicted by thresholding:

if 𝑦^ ≥ 0.5 then class 1, else class 0 if y^ ≥0.5 then class 1, else class 0

To train the model we `minimize binary cross entropy loss` 
$$L(y, y^) = -y log(y^) - (1-y) log(1 - y^)$$


*Training* is typically done using gradient descent:
 - Compute loss on all data
 - Compute gradients of the loss w.r.t. weights
 - Update weights in the opposite direction of the gradient


When you have more than two classes:
Use softmax regression (a.k.a. multinomial logistic regression)
It generalizes sigmoid to multiple classes by using the softmax function to produce a probability distribution

For the logistic regrssion classifier the paramters are usually estimeted by maximizing the log-likelyhood, or equivalently minimizing the binary cross-entropy.
With maximum likelyhood estimation, we chose the parameters which maximize the probability of the observed labels.

We optimize the log-likelihood of the observed class labels given the input data and the model parameters.

---


### Limitations
Only models linear decision boundaries
Can't naturally capture complex, non-linear patterns
Not great for high-dimensional or unstructured data (e.g., images)


---


If you use neural networks, you don't use logistic regression

You're doing classification with a neural network that uses softmax at the output layer.

Extending softmax logistic regression to use a neural network instead of just linear functions provides one huge benefit:
 
Non-linearity and Expressive Power
Logistic regression (softmax with linear inputs):
z = w * x + b 

is limited to linear decision boundaries. That means:
It can only separate classes that can be split using straight lines (or intervals in 1D).
It fails when the data is not linearly separable.

With a NN you can fit more complex patterns, modeling curve-boundaries, fit XOR logic  and  approximate any function.


Linear softmax regression is simple and fast, great for linearly separable data.
But when data is nonlinear, a neural network lets you learn complex boundaries and gives your model much more flexibility.

Logistic regression is a linear model. 
Softmax regression (a.k.a. multinomial logistic regression) is its multiclass version:
These models are not neural networks. They are linear classifiers — their predictions are based on linear combinations of input features, and their decision boundaries are straight lines (or flat in higher dimensions).


---



P(y=11  |x) = sigma(wTx+b)

Linear decision boundary
Probabilistic output
Convex optimization (global optimum)
Well-calibrated probabilities


The best usecases are where Binary classification with linear separability





Logistic regression is a statistical and machine-learning method used for classification, most commonly for binary outcomes (yes/no, true/false, 0/1).

Logistic regression models the probability that an input belongs to a particular class.

Example questions it answers:

 - Will a customer churn or not?
 - Is an email spam or not spam?
 - Will a transaction be fraudulent or legitimate?

Instead of predicting a number (like linear regression), it predicts a probability between 0 and 1.

---

## How it works (intuition)

It computes a linear combination of input features:

$$z = w0 + w1*x1 + w2*x2$$

That value is passed through the logistic (sigmoid) function:

$$sigma(z) = 1 / (1 + e^-z)$$


The output is interpreted as a probability.

>If probability ≥ threshold (often 0.5) → class 1

Otherwise → class 0

## Why it’s called “regression”

Even though it’s used for classification:
It regresses inputs onto log-odds (logarithm of odds)
The model is linear in parameters, which makes it mathematically convenient

## Key properties

Outputs probabilities (interpretable)
Works well for linearly separable data
Fast to train, low memory usage
Assumes a linear relationship between features and log-odds

Logistic regression is a simple, efficient model that estimates the probability of a categorical outcome using a sigmoid-transformed linear equation.

Regress = fit a model that explains or predicts one variable using others.
Logistic regression fits a linear model where the target variable is the log-odds, not the probability itself.

We assume the training data points are independent:
    D={(x1​,y1​),(x2​,y2​),…,(xN​,yN​)}
The probability of seeing the entire dataset is the product of the probabilities of each data point.

So the likelihood of the dataset given parameters 𝜃 is:
    $$P(D∣θ) = ∏(from i=1 to N) ​P(yi​∣xi​,θ)$$

The probability for a single datapoint is:
    $$P(yi​∣xi​) = (P(y=1∣xi​))^yi * ​(P(y=0∣xi​))^1−yi​$$

Multiplying many probability might cause data underflow  and hard optimization 
so taking the log, these are summed.
    $$logP(D∣θ)=i=1∑N​logP(yi​∣xi​)$$


$$logP(D∣θ)=i=1∑N​logP(yi​∣xi​)$$

by substituting this: $$P(yi​∣xi​) = (P(y=1∣xi​))^yi * ​(P(y=0∣xi​))^1−yi​$$

we get:

$$logP(D∣θ) = ∑(from i=1 to N)​[yi​logpi​+(1−yi​)log(1−pi​)]$$

we get the log-likelyhood (binary cross entropy):

$$Loss=−i=1∑N​[yi​logpi​+(1−yi​)log(1−pi​)]$$






Logistic regrsssion can be thought as a single neuron with a sigmoid


Logistic regression fits a linear model where the target variable is the log-odds, not the probability itself.

What logistic regression actually models

Logistic regression does not say:

𝑝 = w0 +  w1* x1 + w2 * x2 +   ...

instead it says: 

log(p/(1-p)) = w0 +  w1* x1 + w2 * x2 +   ...

THE LINEAR PART IS  ON THE LOG-ODDS SCALE NOT THE PROBABILITY SCALE.

Linearity becomes reasonable:

Log-odds can take any real value
Linear combinations make sense on this scale
Probabilities themselves cannot be linear they’re bounded between 0 and 1
The sigmoid is just the inverse step

After fitting a linear model for log-odds, logistic regression converts it back to a probability:

p = 1 / (1 + e^-(w0 + w1*x1 + w2*x2 + ...))

So:
Linear model → log-odds → Sigmoid → probability

Each coefficient wi
Is a constant change in log-odds

Corresponds to a multiplicative change in odds

Logistic regression fits a straight line in log-odds space, then bends that line into an S-shaped curve to produce valid probabilities.




Logistic regression is best understood not as a general-purpose machine-learning tool, but as a very precise instrument for a specific class of problems. It excels when the task is classification and the relationship between inputs and outcomes is relatively simple, stable, and interpretable.

At its core, logistic regression models the probability of an event happening—such as a customer churning, a transaction being fraudulent, or a patient having a disease—as a smooth function of the input features. What makes it powerful is that this probability has a clear mathematical meaning and is directly usable for decision-making. The output is not just a label, but a calibrated likelihood, which is crucial in domains where decisions involve costs, risks, or thresholds.

One of the main reasons logistic regression remains widely used is interpretability. Each input feature contributes additively to the log-odds of the outcome, and each coefficient can be interpreted as an odds ratio. This makes the model transparent: you can explain why a prediction was made, which variables increased or decreased the risk, and by how much. In fields such as healthcare, finance, law, or public policy, this transparency is not optional—it is often a requirement.

Logistic regression is also particularly effective when data is limited or moderately sized. Because its optimization problem is convex, training is stable and fast, and the model reliably converges to a single global solution. This makes it robust, reproducible, and easy to debug. In practice, it often serves as the first model trained on a new problem, acting as a sanity check for data quality and feature relevance. If logistic regression performs well, it is a strong signal that the underlying problem structure is simple and well captured by the available features.

Another important strength lies in probability calibration. Logistic regression tends to produce probabilities that correspond well to observed frequencies, which is not always true for more complex models. This makes it especially valuable in risk estimation tasks such as credit scoring, medical diagnosis, or fraud detection, where the decision threshold may change depending on business or safety constraints.

However, logistic regression is not designed to handle everything. It struggles when the true relationship between inputs and outputs is highly nonlinear, when interactions between features are complex, or when the data has a temporal or sequential structure. It is also unsuitable for raw unstructured data such as images, audio, or text unless significant feature engineering is applied beforehand. In those cases, more expressive models are needed.

From a systems perspective, logistic regression works best when the problem is essentially static and memoryless: each decision depends only on the current inputs, not on past states or long-term dynamics. When this assumption holds, logistic regression is not just adequate—it is often optimal.

In summary, logistic regression is best used when you need a simple, interpretable, well-calibrated classifier that works reliably with limited data and produces probabilities you can trust. It may not be the most sophisticated model, but in the right context, it is often the most appropriate and defensible choice.