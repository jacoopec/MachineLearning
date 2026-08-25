import numpy as np
from scipy import stats

data = np.random.randint(-20, 61, size=20)

mean = np.mean(data)
mode = stats.mode(data, keepdims=True) 
median = np.median(data)
sorted_data =  np.sort(data)

variance = np.var(data)
stdDeviation = np.std(data)

print(sorted_data)
print("Mean ",mean," mode: ",mode[0]," median: ",median," variance: ",variance," std. deviation: ",stdDeviation)
