import math

#Split valuation of decision trees (ID3 / C4.5)
y=[0,0,1,0,0,0,1,1,1]

def entropy(labels):
    if len(labels) == 0:
        return 0

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    ent = 0
    total = len(labels)
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent


def best_entropy_split(y):
    parent_entropy = entropy(y)
    n = len(y)

    best_gain = -1
    best_index = None

    for i in range(0, n):  # valid split points
        left = y[:i]
        right = y[i:]

        weighted_entropy = (
            len(left) / n * entropy(left)
            + len(right) / n * entropy(right)
        )

        info_gain = parent_entropy - weighted_entropy

        if info_gain > best_gain:
            best_gain = info_gain
            best_index = i

    return best_index, best_gain

def split_array(arr, index):
    if not isinstance(index, int):
        raise TypeError("index must be an integer")

    if index < 0 or index > len(arr):
        raise ValueError("index out of range")

    left = arr[:index]
    right = arr[index:]

    return left, right


# Example

index, gain = best_entropy_split(y)

split1, split2 = split_array(y, index)

# print("Best split index:", index)
# print("Information gain:", gain)

print(split1)
print(split2)

index1, gain = best_entropy_split(split1)
index2, gain = best_entropy_split(split2)

print(index1)
print(index2)


split1, split2 = split_array(y, index1)