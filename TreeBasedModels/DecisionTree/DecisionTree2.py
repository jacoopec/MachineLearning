import numpy as np

data = np.genfromtxt("beachvolley_data_augmented.txt", dtype=str, skip_header=1)
i = 0
print(data)
for i in range(data.shape[0]):
    if data[i, 3] == "Yes":
        print(f"Row {i} we can play")

