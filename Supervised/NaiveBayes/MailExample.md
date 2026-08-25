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

