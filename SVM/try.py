import numpy as np

Why = np.array([[1,2],[2,3]])
h = np.array([2,2])
by = np.array([0,0])

y = Why @ h + by
print(y)