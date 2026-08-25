import numpy as np

dataset = np.arange(100)
n = 10

sample = np.random.choice(dataset, size=n, replace=False)

print(sample)