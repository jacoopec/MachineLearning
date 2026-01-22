import random
from itertools import islice

def bagging(data):
    ri = random.randint(1, 4)

    items = list(data.keys())
    random.shuffle(items)

    return items[:ri]
    

dict  = {
    "Sunny":          [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    "Windy":          [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
    "Rainy":          [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
    "HotTemperature": [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
}
print(bagging(dict))