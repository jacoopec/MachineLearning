import random

# Dataset with N elements
dataset = list(range(100))  # example: N = 100

n = 10  # number of elements to sample

# Sample n different elements without replacement
sample = random.sample(dataset, n)

print(sample)