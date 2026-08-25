# ROC curve
The ROC curve `Receiver Operating Characteristic curve` is a way to evaluate how well a binary classifier separates two classes, across all possible decision thresholds.
It helps you visualize how well your model performs across different decision thresholds, balancing true positives against false positives.

the diagonal line: $TPR = FPR$ represents random guessing:
 - Above the diagonal → better than random
 - Below the diagonal → worse than random (flip predictions)

## Area under the curve
The AUC is the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative one.
 - AUC = 1.0 → perfect classifier
 - AUC = 0.5 → random guessing
 - AUC < 0.5 → systematically wrong

The ROC curve tells you how well your model separates classes, regardless of threshold, by showing the trade-off between catching positives and raising false alarms.