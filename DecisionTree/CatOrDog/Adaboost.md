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