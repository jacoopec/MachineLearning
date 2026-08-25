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
numeric = df.select_dtypes(include='number')
correlation = numeric.corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()




