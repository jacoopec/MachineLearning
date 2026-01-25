The ROC curve (Receiver Operating Characteristic curve) is a way to evaluate how well a binary classifier separates two classes, across all possible decision thresholds.
The ROC curve (short for Receiver Operating Characteristic curve) is a super useful tool in machine learning, especially for evaluating binary classifiers—like deciding if an email is spam or not, or if a medical test is positive/negative. It helps you visualize how well your model performs across different decision thresholds, balancing true positives against false positives.

he diagonal line:
TPR = FPR

-Represents random guessing.
-Above the diagonal → better than random
-Below the diagonal → worse than random (flip predictions)

AUC = 1.0 → perfect classifier
AUC = 0.5 → random guessing
AUC < 0.5 → systematically wrong

AUC = probability that a randomly chosen positive example is ranked higher than a randomly chosen negative one.

The ROC curve tells you how well your model separates classes, regardless of threshold, by showing the trade-off between catching positives and raising false alarms.