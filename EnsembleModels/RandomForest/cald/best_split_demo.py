import numpy as np


# Tabular data with 2 features and n rows.
data = np.array([
    [1.0, 2.0],
    [2.5, 3.0],
    [3.0, 4.0],
    [2.0, 3.0],
    [3.0, 1.0],
    [8.0, 5.0],
    [6.0, 6.0],
    [9.0, 6.0],
    [7.0, 8.0],
    [1.06094342, 1.79200318],
    [2.65009024, 3.18811294],
    [2.60979296, 3.7395641],
    [2.02556808, 2.93675148],
    [2.99663977, 0.82939121],
    [8.17587959, 5.15555839],
    [6.01320614, 6.22544824],
    [9.09350187, 5.82814151],
    [7.07375016, 7.80822348],
    [1.17569006, 1.99001482],
    [2.46302753, 2.86381409],
    [3.24450827, 3.9690941],
    [1.91433444, 2.92957329],
    [3.10646184, 1.07308881],
    [8.08254652, 5.0861642],
    [6.42832952, 5.918717],
    [8.89755145, 5.83724545],
    [7.12319588, 8.22579446],
    [0.97721051, 1.8319687],
    [2.33510376, 3.13011856],
    [3.14865083, 4.10863085],
    [1.86689806, 3.04643226],
    [3.02333716, 1.04373772],
    [8.17428576, 5.04471911],
    [6.13578271, 6.01351581],
    [9.05782388, 6.12625765],
    [6.70856884, 7.93606576],
])

labels = np.array([
    0, 0, 1, 0, 0, 1, 0, 1, 1,
    0, 0, 1, 0, 0, 1, 0, 1, 1,
    0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 1, 1, 0, 0, 1, 0, 0, 0,
])


def gini_impurity(split_labels):
    """Return the Gini impurity of a group of labels."""
    if len(split_labels) == 0:
        return 0

    _, counts = np.unique(split_labels, return_counts=True)
    probabilities = counts / len(split_labels)

    return 1 - np.sum(probabilities**2)


def weighted_gini(left_labels, right_labels):
    """Return weighted Gini impurity after splitting the labels."""
    total = len(left_labels) + len(right_labels)
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total

    return (
        left_weight * gini_impurity(left_labels)
        + right_weight * gini_impurity(right_labels)
    )


def find_best_split(data, labels, feature_index):
    """
    Find the best threshold for one feature.

    Args:
        data: NumPy array with shape (n_rows, 2).
        labels: NumPy array with shape (n_rows,).
        feature_index: Index of the feature to split on, for example 0 or 1.

    Returns:
        best_threshold, left_split, right_split
    """
    if len(data) == 0:
        raise ValueError("data must contain at least one row")

    if len(data) != len(labels):
        raise ValueError("data and labels must have the same number of rows")

    if feature_index < 0 or feature_index >= data.shape[1]:
        raise ValueError(f"feature_index must be between 0 and {data.shape[1] - 1}")

    values = np.sort(np.unique(data[:, feature_index]))
    if len(values) < 2:
        raise ValueError("selected feature has only one unique value")

    thresholds = (values[:-1] + values[1:]) / 2

    best_threshold = None
    best_left_split = None
    best_right_split = None
    best_score = float("inf")

    for threshold in thresholds:
        left_mask = data[:, feature_index] <= threshold
        right_mask = data[:, feature_index] > threshold

        left_labels = labels[left_mask]
        right_labels = labels[right_mask]
        score = weighted_gini(left_labels, right_labels)

        if score < best_score:
            best_score = score
            best_threshold = threshold
            best_left_split = data[left_mask]
            best_right_split = data[right_mask]

    return best_threshold, best_left_split, best_right_split


if __name__ == "__main__":
    feature_index = 0
    
    n = 4  # number of rows to pick

    idx = np.random.choice(data.shape[0], size=n, replace=False)

    subset_data   = data[idx]
    subset_labels = labels[idx]
    
    threshold, left_split, right_split = find_best_split(subset_data, subset_labels, feature_index)

    print(f"Best split for feature {feature_index}")
    print(f"Threshold: {threshold}")
    # print("\nLeft split:")
    # print(left_split)
    # print("\nRight split:")
    # print(right_split)
    print(f"data shape: {data.shape},\n right split shape: {right_split.shape},\n left split shape: {left_split.shape},\n percentage in left split: {len(left_split) / len(data):.2f},\n percentage in right split: {len(right_split) / len(data):.2f}  ")
