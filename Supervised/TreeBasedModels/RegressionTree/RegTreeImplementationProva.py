import numpy as np
import matplotlib.pyplot as plt

dosage = np.array([
     1,  3,  5,  7,  9, 11,
    13,
    15, 17, 19, 21,
    26, 27, 28, 29, 30,
    30, 32, 34, 36
])

effectiveness = np.array([
     2,  2,  2,  2,  3,  5,
    20,
   100, 100, 100, 100,
    65, 60, 55, 50, 48,
    12,  2,  2,  2
])

sum_of_sqrd_res = []
means = []

for x in range(dosage.size):
    right_vals = effectiveness[x:dosage.size]
    left_vals = effectiveness[0:x]
    mean_right =  np.mean(right_vals)
    mean_left  =  np.mean(left_vals)
    
    vals = np.concatenate((left_vals - mean_left, right_vals - mean_right))
    res = (vals)**2
    sum_of_sqrd_res.append(res.sum())
    
    
print(means)
print(sum_of_sqrd_res)
plt.scatter(dosage, sum_of_sqrd_res)
plt.xlabel("X values")
plt.yscale('linear') 
plt.ylim(1, 30000)
plt.ylabel("Y values")
plt.title("Simple Scatter Plot")
plt.show()