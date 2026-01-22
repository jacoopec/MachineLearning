import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 🔧 1. Create a synthetic dataset
data = {
    'SquareFeet': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000,3100,3300,3600,3650],
    'Price':      [100,  120,  130,  155,  180,  200,  210,  240,  260,   290, 295,322,320,325]  # in thousands
}
df = pd.DataFrame(data)

# 📊 2. Prepare features and labels
X = df[['SquareFeet']]
y = df['Price']

# 🔁 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 4. Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 5. Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# 🖼️ 6. Visualize the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Square Feet')
plt.ylabel('Price (thousands)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
