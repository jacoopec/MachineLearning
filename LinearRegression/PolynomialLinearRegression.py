import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

#Tried overfitting, by increasing the degree of the polynomial function.
#training score is very high (e.g. R² ≈ 1)	Model fits training data perfectly
#Test/validation score is low	Model fails on new data — poor generalization
#Curve looks too wiggly	It memorizes points instead of modeling trends

# Create a synthetic nonlinear dataset
# y = 0.5x³ - 2x² + x + noise
np.random.seed(82)
X = np.linspace(-9, 9, 30).reshape(-1, 1)
y = 0.5 * X**3 - 2 * X**2 + X + 20*np.random.randn(*X.shape)

# Build a pipeline: PolynomialFeatures + LinearRegression
degree = 15
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

# Prediction
X_test = np.linspace(-9, 9, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Evaluation
y_train_pred = model.predict(X)
mse = mean_squared_error(y, y_train_pred)
r2 = r2_score(y, y_train_pred)
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# Plotting
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label=f'Degree {degree} Fit')
plt.title(f'Polynomial Regression (degree {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
