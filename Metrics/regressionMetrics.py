import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def MAE(y_new_true,y_pred):
    # print(y_new_true)
    print(y_pred)
    print(abs(sum(y_new_true - y_pred)))
    return 1/(len(y_new_true)) * abs(sum(y_new_true - y_pred))

# Training data
X = np.array([[1], [2], [3], [4], [5]])   # feature values
y = np.array([5, 8, 11, 14, 17])          # target values (3x + 2)

# Create and train model
model = LinearRegression()
model.fit(X, y)



# Results
# print("Slope (coefficient):", model.coef_[0])
# print("Intercept:", model.intercept_)

# Predict for a new value
x_new = np.array([[6],[20]])
y_pred = model.predict(x_new)
y_new_true = 3*x_new +  2
y_new_true = y_new_true.squeeze()

mse = mean_squared_error(y_new_true, y_pred)

print("mse")
print(mse)

print(MAE(y_new_true, y_pred))
# print("Prediction for x=6:", y_pred[0])
