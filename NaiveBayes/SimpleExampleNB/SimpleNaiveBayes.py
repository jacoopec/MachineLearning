import pandas as pd

#Classufy an email as spam based on the words
#40 are Spam, 60 not Spam
#Cheap appears in 30 spam emails and in 5 not spam emails
data = {
    "Emails":          [40, 60],
    "cheapWord":   [30, 5],
    "meetingWord": [2, 25],
    "moneyWord":   [80, 15],
    "jobWord":     [23, 28]
}

df = pd.DataFrame(
    data,
    index=["Spam", "Not Spam"]
)

df['priors'] = (df['Emails'][0]/(df['Emails'][0]+df['Emails'][1]),df['Emails'][1]/(df['Emails'][0]+df['Emails'][1]))
df['cheap'] = (df['cheapWord'][0]/df['Emails'][0],df['cheapWord'][0]/df['Emails'][1])
df['meeting'] = (df['meetingWord'][0]/df['Emails'][0],df['meetingWord'][0]/df['Emails'][1])
df['money'] = (df['moneyWord'][0]/df['Emails'][0],df['moneyWord'][0]/df['Emails'][1])
df['job'] = (df['jobWord'][0]/df['Emails'][0],df['jobWord'][0]/df['Emails'][1])
print(df)

#new word containing cheap, meeting, moeny and job
spam = df['priors'][0]* df['cheap'][0]*df['meeting'][0]*df['money'][0]*df['job'][0]
notSpam = df['priors'][1]* df['cheap'][1]*df['meeting'][1]*df['money'][1]*df['job'][1]

print(spam)
print(notSpam)

# Easy things you can do:
# print(df.at["Spam", "cheap appears"])          # → 30
# print(df["cheap appears"] / df["Emails"])       # → spam rate of "cheap"
# print(df.sum())   


