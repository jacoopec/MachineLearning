Input features:
Income
Debt-to-income ratio
Credit utilization
Number of late payments
Loan amount
Employment length
Age

In credit risk, the true relationship is often well-approximated by:
log((P(default))/(1-P(Default))) = w1 * DebtToIncome + w2*latePayments + w3 * Age + ...

This is exactly the hypothesis class of logistic regression.
Deep models add expressive power that the data does not require.
Deep learning overfits subtle noise

Deep networks:
pick up spurious correlations
memorize cohort-specific patterns
amplify dataset biases

Logistic regression:
has a strong inductive bias
ignores weak nonlinearities
focuses on dominant signals

In out-of-time validation (future customers), logistic regression often has lower error drift.


When the true signal is linear, stable, and interpretable, logistic regression doesn’t just compete with deep learning — it beats it.


