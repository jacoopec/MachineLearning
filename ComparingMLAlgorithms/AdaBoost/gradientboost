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
