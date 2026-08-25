import numpy as np
import matplotlib.pyplot as plt

def compute_gradients(y,weights,bias,x):
    m = x.shape[0]
    gradients = -1/m  * np.sum(y - np.exp(weights @ x.T)/(1 + np.exp(weights @ x.T)))*x
    return gradients 

def computing_loss(weights, bias, x, y):
    m = x.shape[0]
    P0 = (np.exp(weights @ x.T )) / (1 + np.exp(weights @ x.T))
    P1 = 1 - P0
    loss = -1/m * np.sum(y * np.log(P0) + (1-y) * np.log(P1))
    # loss = -1/m * np.sum(y*weights.T @ x -  np.log(1 + np.exp(weights.T@x)))
    # print(y)
    # print(weights)
    # print(x)
    # loss = -(1 / N) * np.sum(Y * (X @ w) - np.log(1 + np.exp(X @ w)))
    # loss = -1/m * np.sum(y*(weights @ x.T) - np.log(1 + np.exp(weights @ x.T)))

    return loss

def predict(w, b, x):
    P0 = (np.exp(w @ x.T + b)) / (1 + np.exp(w @ x.T + b))
    P1 =                    1 / (1 + np.exp(w @ x.T + b))
    return P0, P1
      

weights = np.array([[0.47088303,1.89417661]])#,10,0.3,0.4]])
b = np.array([0.1,0.3,0.2,0.1])
x = np.array([[1,0.2],[2,0.4],[3,0.6],[4,0.8]])
y = np.array([[1],[1],[0],[0]])

print(computing_loss(weights, b, x, y))
alfa = 0.001
loss = 1
iterations = 0

# while abs(loss) > 0.5:

print_weights = False
predictions = False

# np.sum(y - np.exp(weights @ x.T)/(1 + np.exp(weights @ x.T)))*x
print(np.exp(weights @ x.T))
print(np.sum(np.exp(weights @ x.T)))

# for i in range(1000):
#     loss  = computing_loss(weights, b, x, y)
#     grads = compute_gradients(y, weights, b, x)
#     weights = weights - alfa * grads
#     iterations += 1
#     P0, P1 = predict(weights, b, x)
#     print(f"Iteration: {iterations}, Loss: {loss}")#, predictions: {P0},{P1}")
#     if (print_weights):
#         print(f"Weights: {weights}")
#     if(predictions):
#         print(f", predictions: {P0}")
print(f"Final Weights: {weights} {weights.shape}")

