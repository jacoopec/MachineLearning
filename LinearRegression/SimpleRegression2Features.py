import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Create dataset with 2 features
data = {
    'SquareFeet': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000],
    'Rooms':      [2, 2, 3, 3, 4, 4, 4, 5, 5, 6],
    'Price':      [100, 120, 130, 155, 180, 200, 210, 240, 260, 290]
}
df = pd.DataFrame(data)

# Features and label
X = df[['SquareFeet', 'Rooms']]
y = df['Price']

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#  Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot actual data points
ax.scatter(df['SquareFeet'], df['Rooms'], df['Price'], color='blue', label='Actual')

# Create grid for surface plot
x_surf, y_surf = np.meshgrid(
    np.linspace(df['SquareFeet'].min(), df['SquareFeet'].max(), 10),
    np.linspace(df['Rooms'].min(), df['Rooms'].max(), 10)
)
z_surf = model.predict(np.column_stack((x_surf.ravel(), y_surf.ravel()))).reshape(x_surf.shape)

# Plot prediction surface
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='red')

ax.set_xlabel('Square Feet')
ax.set_ylabel('Rooms')
ax.set_zlabel('Price')
plt.title("Linear Regression: 2 Features")
plt.legend()
plt.tight_layout()
plt.show()
