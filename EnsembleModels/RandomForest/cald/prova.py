import numpy as np


j = np.array([1,1])

def function():
    global j
    j += 1
    

if __name__ == "__main__":
    function()
    print(j)