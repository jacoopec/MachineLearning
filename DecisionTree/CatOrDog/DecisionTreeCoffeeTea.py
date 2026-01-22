import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#predict whether a person will like tea or coffee based on:
#Weather: "Sunny", "Rainy"
#Mood: "Happy", "Tired"
#Label: Drink Preference: "Tea", "Coffee"
#What’s Gini Impurity?
#Gini measures how "pure" a node is:
#0 = all samples in one class (pure)
#0.5 = mixed evenly (worst case for binary)
#The tree tries to minimize Gini impurity with each split.

# 🧾 1. Define the dataset
data = {
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Rainy'],
    'Mood': ['Happy', 'Tired', 'Happy', 'Tired', 'Tired', 'Happy', 'Happy', 'Tired'],
    'Drink': ['Tea', 'Coffee', 'Tea', 'Coffee', 'Coffee', 'Tea', 'Tea', 'Coffee']
}
df = pd.DataFrame(data)

# 🎯 2. Convert categorical features to numerical using one-hot encoding or label encoding
df_encoded = pd.get_dummies(df[['Weather', 'Mood']])
label_map = {'Tea': 0, 'Coffee': 1}
y = df['Drink'].map(label_map)

# 📈 3. Train the Decision Tree
model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(df_encoded, y)

# 🧠 4. Visualize the decision tree with Gini index
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=df_encoded.columns,
    class_names=['Tea', 'Coffee'],
    filled=True,
    impurity=True,
    rounded=True
)
plt.title("Decision Tree with Categorical Features")
plt.show()
