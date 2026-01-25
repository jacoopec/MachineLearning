# AdaBoost: An Ensemble Boosting Method

## Overview

Adaboost is an ensembling method.

Adaboost is a boosting method.

As long as a classifier is simple and easy to build and it does a better job than random choice, is a good classifier.

Adaboost combines weak classifier into a good one.

Each learner is going to focus on the weaknesses of the previous one.

By combining weak learners you get a strong learner.

By How much do you scale the mistakes?

How do you combine the classifiers?

With the first weak learner we increase the weight of the wrongly classified points.

The second weak learner 

Each one of the learners is associated with a scaled dataset and thec correct and incorrect score.

Some weak learner has more saying than others, depending on how well it did (how well they learned data)

The sample weight represents how important it is for samples to be correctly classified.

## Comparison to Random Forest

In random forest, each time you make a tree, you make a full sized tree using all the  features.

In a forest of  trees made with adaboost each tree has a single node and 2 leaves.

A forest of stumps  is created.

Stumps use only one variable to make a decision, they are weak learners.

In random forest each tree has a final vote in the final classification.

In a forest of stumps made with adaboost some stumps get more saying than others.

Also the order in which stumps  are created is important.

The errors that the first stump makes influence how the second stump is made.

## Three Main Ideas Behind AdaBoost

three main ideas behind adaboost:

-Adaboost combines a lot of weak learners to make classification.

-Some stumps get more saying in a classification than others. 

-each stumps is made taking the mistakes from the previous stump  into account.

## Creating a Forest of Stumps with AdaBoost

creating a forest of stumps with adaboost

At the beginning, a "sample weight" is given to each sample, this value indicates how important it is to be correctly classified.

At the start each sample get the same weight 1/number of samples

After the first stump is created the sample weights change, in order to create how the next stump is created.

The first stump in the forest is created by finding the variable (feature) that does the best job classifying the sample.

Because all of the samples weight are the same, we can ignore them now.

So taking for instance a specific variable, in the "Yes" branch we have the "correct" number of classified samples which are those that have a "Yes" value and produce a Yes value as classification.

But there are also incorrect classified samples, which are those that even though the feature said "Yes",  produced a final classification of "No".

The  same happens for the "No" branch.

We reapeat this process for all the other features in the dataset.

We calculate the gini index for the 3 stumps, and take the lowest.

Once the stump  is created we have to  find how much saying it will have in the final classification.

This value is determined based on how well it classifis samples.

The total error for a stump is the sum of the weights associated with the incorrectly classified samples.

This stump made 1 error (1 incorrectly classified sample, even the feature said "Yes" the classification is "No")

The sample weights will add up to 1. So the total error will always be between 1 for a horrible stump and 0 for a perfect one.

Total error is used to determine the amount of say this stump has in the final classification.

Amount of say =  1/2 * log((1-total error)/  total error)

when a stump does a good job, and the total error is low, than, the amount of say will be a relatively large number.

When the stump randomly predicts classes, the amount of error is 0.5, so the amount of say will be 0.

A the total error increase, the amount of say a stump has will decrease to numbers lower than 0.

The amount of say is claculated also for the other feature upon which a split could be made.

the total error from each of these is the sum of weights for the incorrectly classified samples.

We now look on the graph the  amount of say value for this error.

Now we need to learn how to modify the weights so that the next stump will take the errors that the current stump made into account.

When we created the first stump, all of the sample weights were the same, so we didn't emphasize the importance of correctly classifying any particular sample.

But since we have an incorrect classification, we will emphasize the need for the next stump  to crrrectly classify it by increasing its sampe weight, and decreasing all of the others because at the end thay have to sum up to 1.

new weight = sample weight * e^amount of say

WHen the amount of say is large, ( the stump did a good job in classification) we will scale the previous sample weight by a large number.

Once the weight is increase, weights have to be normalized.

If we have a weighted Gini function then we use it with the sample  wights, otherwise we use the sample weights to make a new dataset that reflects those weights.


Instead of training one strong model, train many very weak models and combine them intelligently.

A weak learner is a model that performs only slightly better than random guessing.
In practice, this is often:

a decision stump (a tree with a single split)

AdaBoost works by:

Training a weak learner
Focusing more on the samples it got wrong
Training the next learner on this harder dataset
Repeating
Combining all learners into a weighted vote