# Gradient Boosting: An In-Depth Guide


Gradient Boosting is a powerful *ensemble machine learning technique* that builds predictive models sequentially by combining multiple weak learners (typically decision trees) to create a strong predictor. 
It focuses on **minimizing errors from previous models by fitting new trees to the residuals (errors) of the ensemble**. This method excels in both regression and classification tasks, handling non-linear relationships, feature interactions, and complex patterns without requiring feature scaling.

*Gradient Boosting performs gradient descent in function space, where each new tree approximates the negative gradient of the loss function to correct prior mistakes.*
The `gradient` in the name comes from this process: **residuals represent gradients of the loss**, and trees act as basis functions in the optimization.


## Why Use Decision Trees in Gradient Boosting?

Decision trees are the default base learners in Gradient Boosting due to their strengths:

- **Handle non-linear relationships**: Trees can capture complex, non-linear patterns in data without assumptions of linearity.
- **Handle feature interactions automatically**: Splits consider interactions between features naturally.
- **Require no feature scaling**: Trees are invariant to monotonic transformations of features.
- **Can fit complex residual patterns**: They adapt well to the errors (residuals) from previous models.

Gradient Boosting builds trees sequentially, where each new tree learns to correct the mistakes of the current model by fitting the negative gradient of the loss function.

Gradient Boosting builds a strong predictor by iteratively adding small decision trees that follow the gradient of the loss function to correct previous errors.

**Update the model**: Add the new tree to the model:  

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$  

Where \eta is the learning rate (shrinkage factor), and h_m(x) is the new tree.

**Why “gradient” boosting?**  
- The residuals are gradients of the loss.  
- The algorithm performs gradient descent in function space.  
- Trees are the basis functions.

## How Gradient Boosting Fits a Model (Regression)

When predicting a continuous value (e.g., weight), Gradient Boosting starts with an initial guess and iteratively improves it by focusing on errors.

1. **Initial Prediction**: The first guess is the average value of the target variable. If we stop here, this mean would be the prediction for all samples.

2. **Compute Residuals (Errors)**: The errors from the previous model are the differences between observed values and predictions (e.g., observed weights minus the mean). Like AdaBoost, each tree is based on the errors made by the previous tree.

3. **Build and Scale Trees**:  
   - Build a fixed-size tree (can be larger than a stump, unlike some AdaBoost variants) on the residuals.  
   - If multiple samples end up in the same leaf, replace them with their averages.  
   - Scale the tree's predictions by the learning rate (shrinkage). Unlike AdaBoost, all trees are scaled by the same amount.  
   - Add this scaled tree to the ensemble: Combine the average with the new tree built on residuals. Adding the scaled residuals to the previous prediction gets closer to the original values.

4. **Iterate**:  
   - Continue building trees based on the updated residuals (errors from the combined model).  
   - Each time a tree is added, residuals get smaller.  
   - Stop when the specified number of trees is reached, or additional trees fail to improve the fit (reduce residuals).

5. **Final Prediction**: A weighted sum of all trees' outputs. The contribution of each tree is scaled by the learning rate to control the model's complexity and prevent overfitting.

**Loss Function for Regression**:  
$$ L(y, F(x)) = \frac{1}{2} (y - F(x))^2 $$  
Where $ F(x) $ is the model's prediction function. This is the mean squared error (MSE), and residuals are its negative gradients.

Gradient Boost builds fixed-size trees based on previous errors, scales them uniformly, and continues until convergence or the maximum number of trees.

## Gradient Boosting for Classification

For classification, Gradient Boosting adapts the process to handle probabilities and class labels.

1. **Initial Prediction**: The initial guess is the log(odds) of the classes, akin to the logistic regression equivalent of the average in regression.

2. **Convert to Probabilities**: Just like in logistic regression, convert log(odds) to probabilities using the logistic (sigmoid) function for classification.

3. **Compute Residuals**: Residuals are based on the differences between observed labels (one-hot encoded or similar) and predicted probabilities, scaled by the gradient of the loss (e.g., cross-entropy).

4. **Build and Scale Trees**: Similar to regression—build trees on residuals, scale by the learning rate, and add to the ensemble. Trees correct classification errors sequentially.

5. **Iterate and Final Prediction**: Continue adding trees until the stopping criteria. The final output is the class with the highest probability from the summed log(odds).

Gradient Boosting for classification follows the same sequential error-correction but uses a loss like multinomial deviance or cross-entropy.

## Comparison to AdaBoost

- **Similarities**: Both build trees based on previous errors and scale contributions.  
- **Differences**:  
  - Gradient Boosting allows larger trees (not limited to stumps).  
  - Scaling is uniform across trees (same learning rate), unlike AdaBoost's adaptive weights.  
  - Focuses on gradients of any differentiable loss, making it more flexible.

## Implementation Tips

- **Libraries**: Use XGBoost, LightGBM, or scikit-learn's `GradientBoostingClassifier/Regressor`.  
- **Hyperparameters**: Tune `n_estimators` (number of trees), `learning_rate` (shrinkage), `max_depth` (tree complexity).  
- **Best Practices**: Use early stopping to avoid overfitting; monitor validation loss.













---
It starts making a single leaf instead of a tree or stump like adaboost.
This leaf represents an initial guess for the amount to predict
the first guess is an everage value.
This tree is based on the errors made by the previous tree.
Gradient boost, like adaboost builds trees based on the previous 
tree's errors. Unlike adaboost, they are not stumps.
Gradient Boost scales trees by the same amount.
the errors are the differences between the observed and predicted value.

Gradient boost uses a learning rate to scale the contribution from the new tree 

taking lots of small steps in the right direction results in better 
predictions with the testset, lowering the variance.


What is Gradient Boosting?
Gradient Boosting is a machine learning technique for building an ensemble of weak learners (usually decision trees) sequentially, where each new model tries to correct the errors of the previous one.
It's like this:
Instead of learning everything at once, the model learns step-by-step, improving itself a little at each stage.



Let’s say we’re doing regression (predicting a number). Here's how gradient boosting works:

1. Start with a simple prediction
Usually, just the mean of the target values.

2. Calculate errors (residuals)
For each sample:

residual = 𝑦true − 𝑦predicted

​
3. Fit a small tree to predict the residuals
This tree learns what the original model got wrong.

4. Update the model
Add the tree’s predictions to the original prediction:

new prediction=old prediction+η⋅tree output
η (eta) is the learning rate.

5. Repeat
Keep adding new trees that correct previous errors.


Why It's Called "Gradient" Boosting
Each step is like taking a gradient descent step to minimize the loss function (like Mean Squared Error).
The trees try to predict the negative gradient of the loss — in other words, the direction to move to reduce error.


| Step | Purpose                  | Action                                              |
| ---- | ------------------------ | --------------------------------------------------- |
| 1    | Initialize model         | Start with average prediction (log-odds)            |
| 2    | Compute residuals        | Based on how wrong current prediction is            |
| 3    | Fit tree to residuals    | Tree tries to correct errors                        |
| 4    | Update model predictions | Add small change (learning rate × tree output)      |
| 5    | Repeat steps 2–4         | Gradually reduce error                              |
| 6    | Final prediction         | Sum of all tree corrections, passed through sigmoid |






Why not making a tree as long as it needs to be to get a model that perfectly splits the data?
If you have a dataset that is super representative of the population that you are studying is not a bad idea.
Otherwise it might make so many splits that it learns all the quirks of the dataset used, and fail to fit 
additional data. This is overfitting and is a problem  that arise when the model has too much capacity.
In general the capacity is reduced by limiting the number of splits

A popular question in ML is: are  more models better than fewer models?
model ensambling: how to construct aggregations of models that improves test accuracy while reducing  with the cost of storing
training and
ensembling models  applied to decision trees:
-random forest. the basic building blocks of RF are decision trees.
decision trees perform classification or regression by recursively asking simple true or false quesstions that split the data into the purest possible subgroups.
In RF we create a forest of decision trees and take a vote among the  different trees.
in classification, each tree splits out a class prediction and the class with most votes becomes the ouput of the RF.
In regression, the average of each predictions is the output.
the main idea is that there is wisdom in crowds.

Insights from a large group of model is more  likely to be accurate than the prediction from any model alone.
the trees are different, they disagree on what the splits are and what the decisions are.
A large group of uncorrelated trees, working together in an ensemble will outperform any single tree.
to get this uncorrelatedness we use:
-bootstrapping
creating smaller datasets
we allow each tree to randomly sample a subset with replacement.
-bootstrap  aggregating is bagging.
-feature randomness shuffles and take only few features.
A decision tree, when it is time to split data, it consider every features.
In random  forest there are only few features to choose from.
RF builds independent decision trees and combine them

gradient boosted trees use BOOSTING : weak learners are combined sequentially so that each new tree corrects the error  of the previous one.
the first step is to fit a single decision tree. evaluating how well it does with a loss function.
Now a new tree is ceated, and this will be added to the first and it has to lower the loss.
the direction is found deriving the loss by the previous model's output.
This previous output is F(1)..
For any step m, a gradient boosted  tree produces a model such that:

ensamble at step m  
    F(m)                = F(m - 1) + learningrate * weak learner at step m-1. dL/dF(m-1)

compared to random forest, gradient boosted trees have a large model capacity,  so that they can model complex relationship and decision boundariees 


DL
The problem of supervised  learning can also be expressed as the prob of function approximation or curve fitting.
a  model with expressivness can express a wide rangge of functions.
A set of models that can fit complex functions are neural networks.
A NN is a sequential  arrengements of logistic  regression functions whose ouputs are then fed as inputs to other neurons.
This arrengments of simple functions is capable of representing complex functions.
The XOR function can be represented as a composite function, which takes the output of other functions as inputs.
F  =  OR(NOT AND(input))

Why is overfitting a problem?
The whole point of ML is to predict the correct result for data the model hasn't been trained on, so,  if the model works well only on data we provided to train it, and outputs bad or  unpprecise results for new data this is not what we wanted to obtain.


---



How gradient boost fits a model to the training data?
When trying to predict a continous value like weight, the first guess is  
the  average value.
like adaboost, this tree is based on the errors made by the previous tree.
Like adaboost, gradient boost builds fixed sizes trees based on the previous 
tree's errors. But unlike adaboost, each tree can be larger than a stump.
Also like adaboost gradient boost scales the trees. but in comparison with adaboost
it scales by the same amount.
then, GB creates another tree based on the errors made by the previous
and then it scales the tree. 
It continous to build trees in this fashion until it has made the number 
of trees you asked for,  or additional trees fail to improve the fit. 
The first prediction is a weighted sum (in regression).
If we would stop now, this mean would be the prediction for all the samples. 
The errors the previous tree made are the differences between the observed weights and the predicted weight, the mean. 
Now, we combine the average computed before with the new tree built on residuals. 
If you now simply add the weight to the residual you get the original values coming from the 
data. 
By scaling the tree you just built with the learning rate, you get predicted values which are  
If multiple samples end up in the same leaf,  we replace them with their averages. 
We add now this new tree to the old one. 
Each time we add a tree to a prediction,  the residual get smaller. 
Then, we keep making trees until we reach a maximum or adding additional  trees 
doesn't reduce the size of the residuals. 
We scale the tree's contribution to the final prediction witha learning rate. 
We keep adding trees based on the errors made  by the previous tree. 
The loss function used when doing Regression with gradient boost is 
    1/1(observed - predicted)^2 

L(y, F(x)) where F(x) is a function that gives us the predicted values. 


GRADIENT BOOST FOR CLASSIFICATION 
The initial prediction is the log of the odds, which can be thought as the 
logistic regression equivalent of the average.

JUst like in logistic regression, the easiest way to use log(odds) 
for classification is to convert it to a probability, we do that with a 
    logistic function. 
