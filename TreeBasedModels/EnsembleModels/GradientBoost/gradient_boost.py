import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6])
y = np.array([2,3,5,7,5,3])
eta = 1 

F0 = np.sum(y)/y.size
residuals = F0 - y
print(residuals) 

sum_of_sqrd_res = np.array([])
means = np.zeros((1, 2), dtype=int)
weak_learners = np.zeros((1, 2), dtype=int)
tresholds = np.array([])

for i in range(x.size -1):
    right_vals =  residuals[i+1:x.size]
    left_vals  =  residuals[0:i+1]
    mean_right =  np.mean(right_vals)
    mean_left  =  np.mean(left_vals)
    means = np.vstack((means,np.array([mean_right,mean_left])))
    vals = np.concatenate((left_vals - mean_left, right_vals - mean_right))
    res = (vals)**2
    sum_of_sqrd_res = np.append(sum_of_sqrd_res,res.sum())
    
means = np.delete(means, 0, axis=0) 
treshold = np.argmin(sum_of_sqrd_res)
tresholds = np.append(tresholds,treshold)
weak_learners = np.vstack((weak_learners,np.array([means[treshold][0],means[treshold][1]])))
weak_learners = np.delete(weak_learners, 0, axis=0) 

#
residuals = np.array([])
for i in range(x.size):
    if(i < treshold):
        residuals = np.append(residuals,weak_learners[0][0])
    else: 
        residuals = np.append(residuals,weak_learners[0][1])
F1 = F0 + residuals






print(residuals)
print(F1)


# print(sum_of_sqrd_res)
# print(means)
# print(weak_learners)
plt.scatter(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Simple Scatter Plot")
plt.show()