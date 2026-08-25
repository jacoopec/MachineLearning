import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#loading and cleaning
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

url = 'adult/adult.data'
df = pd.read_csv(url, names=columns, header=None, sep=",\s*", engine='python')

# Clean strings
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Replace '?' with NaN and drop rows
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)


# Age distribution
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Income by gender
sns.countplot(data=df, x='sex', hue='income')
plt.title('Income Distribution by Gender')
plt.show()

# Education level
sns.countplot(data=df, y='education', order=df['education'].value_counts().index)
plt.title('Education Levels')
plt.show()

# Hours worked per week
sns.boxplot(data=df, x='income', y='hours-per-week')
plt.title('Hours Worked per Week by Income')
plt.show()





