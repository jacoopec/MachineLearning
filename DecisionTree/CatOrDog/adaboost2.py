import matplotlib.pyplot as plt

X = [0,1,2,3,4]
y = [-1,-1,1,1,-1]
w = [1/5,1/5,1/5,1/5,1/5]

left_err = 0
right_err = 0
errs = [[2,3],[0,0],[0,0],[0,0],[0,0]]


for x in X:
    split = x + 0.5
    i = 0
    while(i <= x):
        errs[i][0] = errs[i][0] + sum(w[0:x])
        errs[i][1] = errs[i][1] + sum(w[x:5])
        i = i + 1

print(errs)

plt.scatter(X, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Simple Scatter Plot")
plt.show()
