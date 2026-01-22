Cross-entropy measures how well a predicted probability distribution q matches the true distribution p.

General (multiclass) cross-entropy
For a true distribution p(x) and predicted distribution q(x):
H(p,q) = −x∑​p(x)logq(x)

predicted probabilities y^, one-hot labels (true class of y)
L=−i=1∑C​yi​log(y^​i​)

Binary cross-entropy
for a single example with label 0 or 1
L=−[ylog(y^​)+(1−y)log(1−y^​)]

H(p,q)​=−[p(1)logq(1)+p(0)logq(0)]=−[ylog(y^​)+(1−y)log(1−y^​)]​

When y=1:
𝐿 = − log(𝑦^)
→ penalizes low predicted probability for the positive class.

When y=0:
L=−log(1−y^)
→ penalizes high predicted probability for the positive class.


Binary cross-entropy is simply cross-entropy between two Bernoulli distributions:
the true (one-hot / deterministic) distribution,
the model’s predicted Bernoulli distribution.
Minimizing it is equivalent to maximum likelihood estimation for a Bernoulli model.