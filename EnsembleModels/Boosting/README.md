# Boosting 

Many weak learners, each performing only slightly better than random, can be combined to form an arbitrarily good ensemble hypothesis.

Boosting starts from a Dataset D and sequentially train equal classifiers fi focusing on errors
from previous classifiers
Assign to every xk ∈ D equal weights wk =1N

Iterate with i from 1 to M number of boosting stages:
1 Sample a new Dataset Di
from D using the weights {wik}N
k=1 as the sampling probability
for every record x
2 train the i-th classifier fi on Di
and measure the accuracy and record the accuracy as αi
3 if xk was wrongly classified augment its weight ate the next stage w
i+1
k
and re-normalize
weights
