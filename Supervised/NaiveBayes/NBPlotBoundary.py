import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Create a sample dataset
data = {
    'Age':    [  18,     20,    22,    25,    47,     52,    46,     47,     56,     57,     55,    60,     61,      62,   61],
    'Income': [  60000, 50000, 15000, 29000, 48000, 60000, 52000,   23000,  65000,  34000 , 70000, 80000, 26000, 82000, 85000],
    'Buys':    [  'No', 'No',  'No',  'Yes', 'Yes', 'Yes',  'No'   ,'Yes',  'No',   'Yes',   'Yes',  'No',  'Yes', 'Yes', "Yes"]
}
df = pd.DataFrame(data)

# Prepare features and label
X = df[['Age', 'Income']].values
y = LabelEncoder().fit_transform(df['Buys'])  # 'No' → 0, 'Yes' → 1

#  Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#  Visualize decision boundary
def plot_decision_boundary(X, y, model, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00AA00'])

    h = 1000  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1,    X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1000, X[:, 1].max() + 1000
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, h),
        np.linspace(y_min, y_max, h)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=80)
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_decision_boundary(X, y, model, "Naive Bayes Decision Boundary")
