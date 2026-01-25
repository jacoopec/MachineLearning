The Naive Bayes classifier is a probabilistic classification algorithm based on Bayes’ theorem.
It’s called naive because it assumes that features are conditionally independent given the class,an assumption that is often false, but surprisingly effective in practice.

It’s widely used for text classification, spam detection, and other high-dimensional problems.

Given the class label, each feature is independent of the others.

The predicted class is the one with the highest posterior probability.

A class prior is just your belief about how common each class is before you look at any features.
“Before I see any data about this example, how likely is each class?”

P(spam) = 0.1
P(notSpam) = 0.9 

These numbers are the class priors.
They reflect the base rate of each class in the world you’re modeling.

TRAINING
Naive Bayes combines three things:

Class prior
“How common is this class overall?”

Estimate feature LikelihoodS
“If the email were spam, how likely would I see these words?”

Posterior
“Given what I see, how likely is this class?”

If spam is very common (70%), the same evidence should push you toward spam much more strongly.

If you treat all classes as equally likely when they are not, your model may:

Overpredict rare classes
Underpredict common classes
Produce misleading probabilities

How this differs from logistic regression
Naive Bayes: priors are explicit and visible
Logistic regression: priors are implicit, absorbed into the intercept term
The concept exists in both, but Naive Bayes makes it obvious.

Limitations of naive bayes classifier

Independence assumption is often unrealistic
Poor probability calibration
Struggles when features are strongly correlated
Decision boundaries are simplistic


Classify an email as Spam or Not Spam based on two words: cheap and meeting
Considering a total of 100 emails:

| Class    | Emails | cheap appears | meeting appears |
| -------- | ------ | ------------- | --------------- |
| Spam     | 40     | 30            | 2               |
| Not Spam | 60     | 5             | 25              |

Priors:
P(Spam) = 0.4
P(not spam) = 0.6

P(cheap|spam) = 30/40  = 0.75
P(meeting|spam) = 2/40 = 0.05
P(cheap|not spam) = 5/60 = 0.083
P(meeeting |not spam) = 25/60 = 0.417

So, classifying a new email which contains "cheap" and "meeting":
spam score = 0.4 * 0.75 * 0.05 = 0.015
not spam score = 0.6 * 0.083 * 0.417 = 0.021



