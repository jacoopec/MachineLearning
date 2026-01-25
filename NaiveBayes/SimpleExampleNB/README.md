Naive Bayes Spam Classifier Example
This is a simple Python script demonstrating a basic Naive Bayes classifier for spam email detection using word counts. It uses pandas to organize data and calculate priors and conditional probabilities.
Description
The script:

Defines data for Spam and Not Spam emails, including counts of specific words ("cheap", "meeting", "money", "job").
Creates a pandas DataFrame with rows for "Spam" and "Not Spam".
Calculates class priors (probability of Spam/Not Spam).
Computes conditional probabilities for each word given the class (e.g., P("cheap" | Spam)).
For a hypothetical new email containing all words ("cheap", "meeting", "money", "job"), it calculates the unnormalized posterior probabilities for Spam and Not Spam by multiplying priors and conditionals (Naive Bayes assumption of independence).
Prints the DataFrame and the computed probabilities.

Note: This is a toy example without normalization (to get actual probabilities summing to 1) or smoothing (to handle zero counts). In real applications, use libraries like scikit-learn for robust implementation.