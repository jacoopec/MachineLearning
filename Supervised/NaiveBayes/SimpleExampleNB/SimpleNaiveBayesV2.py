import pandas as pd

# Classify an email as spam based on word presence probabilities (Bernoulli-like Naive Bayes)

# Data: 40 Spam emails, 60 Not Spam
# Word appearance counts in each class
data = {
    "Emails": [40, 60],
    "cheap_appears": [30, 5],
    "meeting_appears": [2, 25],
    "money_appears": [80, 15],
    "job_appears": [23, 28]
}

df = pd.DataFrame(
    data,
    index=["Spam", "Not Spam"]
)

# Compute priors: P(class)
df['priors'] = df['Emails'] / df['Emails'].sum()

# Compute conditional probabilities: P(word | class) = appearances / num_emails
word_columns = ["cheap_appears", "meeting_appears", "money_appears", "job_appears"]
prob_columns = [col.replace("_appears", "") for col in word_columns]
df[prob_columns] = df[word_columns].div(df['Emails'], axis=0)

print(df)

# Assume a new email contains 'cheap', 'meeting', 'money', and 'job' (each appears once)
# Compute unnormalized posteriors: P(class) * ∏ P(word | class)
spam_posterior = df.loc["Spam", 'priors'] * df.loc["Spam", ['cheap', 'meeting', 'money', 'job']].prod()
not_spam_posterior = df.loc["Not Spam", 'priors'] * df.loc["Not Spam", ['cheap', 'meeting', 'money', 'job']].prod()

print("\nUnnormalized Spam posterior:", spam_posterior)
print("Unnormalized Not Spam posterior:", not_spam_posterior)

# To classify: compare the two (higher wins)
predicted_class = "Spam" if spam_posterior > not_spam_posterior else "Not Spam"
print("Predicted class for the new email:", predicted_class)

# Easy things you can do with the DataFrame:
# print(df.at["Spam", "cheap_appears"])          # → 30
# print(df["cheap"] )                            # → spam rate of "cheap"
# print(df.sum())                                # column totals