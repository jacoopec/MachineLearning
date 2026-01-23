import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("titanic")
df2 = df.isnull().sum()
df["age"] = df["age"].fillna(df["age"].median())
df["deck"] = df["deck"].map({"A": 0, "B": 1,"C":2,"D":3,"E":4,"F":5,"G":6})

print(df["age"])
df["age"].hist()
plt.show()

# 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
#        'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
#        'alive', 'alone'
