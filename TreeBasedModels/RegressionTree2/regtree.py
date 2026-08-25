import matplotlib.pyplot as plt


def computeMeans(x,y):
    means = list(range(len(y)))
    index = 0
    for index in range(len(y)):
        index2 = 0
        for index2 in range(len(y) - index):
            means[index] = means[index] + y[index2 + index]
              
        means[index] = means[index] / len(y)
        
    return means


y = [0.3, 0.1, 0.5, 0, 2, 10, 10, 10, 10,  6.5, 6, 5.5,  5,  1.5, 0,  0]
x = [1, 2, 3, 4, 5,  6,  7,  8,  9,  10, 11,  12, 13, 14, 15, 16]



plt.scatter(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot")

means = computeMeans(x,y)


for i in means:
    plt.axhline(y=means[i], color='r', linestyle='--')   # horizontal line at y = 5
    
plt.show()
