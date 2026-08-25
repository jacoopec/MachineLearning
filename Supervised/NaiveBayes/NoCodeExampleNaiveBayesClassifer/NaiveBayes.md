P(C|X) = P(X|C) *  P(C) / P(X)


C is the class
X is the feature set (Tired, onTime, Mood)

heart of Bayes’ Theorem in classification. Let’s break it down clearly and precisely:

---------------------------------------------------------------

-P(Class∣Features): Posterior probability (what we care about)
This is the probability that an input belongs to a certain class, given the observed feature values. In simpler terms:
“Given this evidence, how likely is each class?”

---------------------------------------------------------------

-P(Features∣Class): Likelihood (from training)
Likelihood is how consistent the data is with each class.
Assuming a certain class is true, how likely is it to see this data?
In Naive Bayes classification, you calculate the likelihood of observing each feature value given a specific class:
P(feature i | class c)
Then, assuming all features are independent (the "naive" part), you multiply the likelihoods:
For example:
Among the cases in which I catch the train, how often am I tired when this happens?
How much it is likely that I'm tired if I catch the train?
In Naive Bayes, we use likelihoods to compare how well each class "explains" the data.
In Naive Bayes likelihoods are multiplied with the prior to compute posterior
| Concept     | Meaning                                                    |
| ----------- | ---------------------------------------------------------- |
| Probability | You know the model (class), asking how likely the data is  |
| Likelihood  | You have the data, asking how likely the class explains it |


---------------------------------------------------------------

-P(Class): Prior (from training)
Prior is what you believed before seeing any data.
It represents your belief about how likely each class is before seeing any evidence or features.
In classification, it's the relative frequency of each class in your training data.
The word “prior” comes from prior knowledge — what you assume about the world before you observe anything new.
In Naive Bayes, it’s literally:
“How common is each class in the dataset?”
Priors weight the final classification decision
If one class is much more common, the model tends to predict it unless the feature evidence strongly supports the rare class

| Term     | Meaning                                              |
| -------- | ---------------------------------------------------- |
| Prior    | Probability of a class **before** seeing features    |
| Based on | Class frequency in training data or domain knowledge |
| Role     | Weights the posterior probability in Naive Bayes     |

Domain knowledge is the understanding, expertise, and insights someone has about a specific field or area (the "domain") where a problem exists.
It's not about the algorithm — it's about the real-world context of the data you're working with.

You might use domain knowledge like:
People who are tired are less alert → more likely to miss the train
Being on time strongly influences catching the train
Mood may correlate with behavior or alertness

Even before seeing any data, this kind of real-world understanding helps you:
Interpret patterns the model finds
Select important variables
Explain results to others

---------------------------------------------------------------

-P(Features): Evidence (normalization factor)
This is the probability of observing the feature combination regardless of the class:
P(Features)= ∑ P(Features∣Class)⋅P(Class)
So it depends on all classes, not any one in particular.
When comparing Class1 (Yes) with class2 (No) is the same, so it can be canceled.
Suppose you want to compare two cars by "value per dollar." If both cost the same price, you can just compare their features directly — price becomes irrelevant in the comparison.


It assumes features are conditionally independent given the class — hence "naive".

------------------------------------

Step 1: Compute Prior Probabilities
How often each class occurs in the training set:

P(yes) = 2/4
P(no) = 2/4

------------------------------------

Step 2: Compute Likelihoods

In Naive Bayes, likelihoods means:
P(feature value | class)  -> MEANING: what is the probability of seeing a feature given a class?


Let's group by the target class CatchTrain? and count the frequencies of each feature value.

------------------------------------
------------------------------------
🟩 1. Class: CatchTrain? = Yes

| Tired | onTime | Mood |
| ----- | ------ | ---- |
| No    | Yes    | Good |
| Yes   | Yes    | Good |

The total samples in this class is 2.
Now count each feature value:

------------------------------------
Tired

I caught the train one time when I was tired and another time when I wasn't.
Tired = Yes → 1 time
Tired = No → 1 time

Ptired(yes|yes) = 1/2
Ptired(no|yes) = 1/2

------------------------------------
onTime 
Yes → 2 times

Everytime, I am on time, I catch the train. If I am late, I never catch the train. 
Pontime(yes|yes) = 2/2 = 1
Pontime(no|yes) = 0

------------------------------------
Mood
Mood = Good → 2 times

The 2 times I caught the train I was in a good mood.
Pmood(good/yes) = 2/2 = 1
Pmood(good/no) = 0

------------------------------------
------------------------------------
🟥 2. Class: CatchTrain? = No

| Tired | onTime | Mood |
| ----- | ------ | ---- |
| Yes   | No     | Bad  |
| No    | No     | Good |


Total samples in this class = 2

------------------------------------
Tired


I caught the train one time when I was tired and another time when I wasn't.
Tired = Yes → 1
Tired = No → 1

Ptired(no|no) = 1/2
Ptired(yes|no) = 1/2

------------------------------------
onTime = No → 2

If I am not on time, I never catch the train.
Pontime(no|no) = 2/2 = 1
Pontime(yes|no) = 0

------------------------------------
Mood

One time that I didn't caught the train I was in a bad mood.
Another time I was in a good mood.

Mood = Good → 1
Mood = Bad → 1

Pmood(good/no) = 1/2 = 0.5
Pmood(bad/no) = 1/2 = 0.5
------------------------------------
For each feature value given a class:

For class = Yes:
Ptired(yes|yes) = 1/2
Ptired(no|yes) = 1/2
Pontime(yes|yes) = 2/2 = 1
Pmood(good/yes) = 2/2 = 1

For class = No:
Ptired(yes|no) = 1/2
Ptired(no|no) = 1/2
Ptired(no|yes) = 1/2
Pontime(no|no) = 2/2 = 1
Pmood(good/no) = 1/2 = 0.5
Pmood(bad/no) = 1/2 = 0.5

------------------------------------

STEP3 MAKE A prediction With this values:
Tired=Yes, onTime=No, Mood=Good

| Tired | onTime | Mood |
| ----- | ------ | ---- |
| Yes   | No     | Good |

cALCULATE THE Probabilities of each class:
for yes:
P(yes) * Ptired(yes | yes) *Pontime(no | yes) * Pmood(good |yes) = 0.5 * 0.5 * 0 * 1

P(no) * Ptired(yes | no) *Pontime(no | no) * Pmood(good |no) = 0.5 * 0.5 * 1 * 0.5


Once you've computed the posterior probability for each class using Bayes’ theorem, you choose the class with the highest posterior probability. That becomes your final prediction.

Since 0.125 > 0 -> prediction = No



Posterior probabilities:

Give not just a decision, but how confident the model is

Can be used in threshold-based classification

Help you build probabilistic decision-making systems


