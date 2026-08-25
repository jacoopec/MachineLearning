#e
import numpy as np
import matplotlib.pyplot as plt

# x = [[ a,b ],[c,d] , [e,f] , [g,h]]
# weights [ a b ]  [ a  c e  g ]   = [ 1 2 3 4 ]
#                    b  d f  h

def compute_gradients(y,weights,bias,x):
    m = x.shape[0]
    # gradients = -1/m  * np.sum(y - weights @ x.T)*x
    # dw = (2 / m) * x.T @ (weights @ x.T - y)
    # print("weights @ x.T")
    # print((weights @ x.T))
    # print("x.T")
    # print(x.T)
    # print("((weights @ x.T).T - y)")
    # print(((weights @ x.T).T - y))
    # print("(weights @ x.T).T - y")
    # print(x.T @ ((weights @ x.T).T - y))
    dw = (2 / m) * (x.T @ ((weights @ x.T ).T +  bias - y))
    # dw = (2 / m) * (x.T @ ((weights @ x.T ).T))
    db = (2 / m) * np.sum((weights @ x.T).T + bias  - y)
    return dw.T, db

def computing_loss(weights, bias, x, y):
    m = x.shape[0]
    # loss = -1/m * np.sum(y*weights.T @ x -  np.log(1 + np.exp(weights.T@x)))
    # print((weights @ x.T).T)
    loss = -1/m * np.sum(( weights @ x.T).T - y)**2
    return loss

weights = np.array([[3.2435896 , 2.44871792]])
bias = np.array([0.1])
x = np.array([[1,0.2],[2,0.4],[3,0.6],[4,0.8]])
y  = np.array([[6.4],[8.8],[11.2],[13.6]])
yy = weights @ x.T
weights_list  = []
loss_list  = []
r = 100
# print((yy.T).shape)

for i in range(3):
    dw, db = compute_gradients(y,weights,bias,x)
    # print(dw,db)
    weights = weights - 0.001 * dw
    bias = bias - 0.01 * db
    loss = computing_loss(weights, bias, x, y)  
    weights_list.append(weights)  
    loss_list.append(loss) 
    print(f"Loss: {loss}")
    print(weights)
    
print(x[:,0])
print(weights)
plt.plot(x[:,0],y)
plt.plot(x[:,1],y)
plt.show()