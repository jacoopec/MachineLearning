#Simple decision tree for visualization using 2 features

import matplotlib.pyplot as plt

data = {
    "Income": [25,30,28,60,65,70,80,40,45,55, 20, 80, 24, 67],
    "Age":    [22,25,28,35,40,45,50,23,30,38, 34, 14, 45,  26],
    "Buy":    [0,  0, 0, 1, 1, 1, 1, 0, 0, 1,  1,  1,  1,  1]
}


def recursive_best_splits(data, splits, min_samples=2):
    # labels
    y = data["Buy"]

    # stop conditions
    if len(set(y)) == 1:
        return

    features = [k for k in data if k != "Buy"]
    if not features or len(y) < min_samples:
        return

    split = best_split(data)
    if split["feature"] is None:
        return

    feature = split["feature"]
    threshold = split["threshold"]

    # store split for later plotting
    splits.append({
        "feature": feature,
        "threshold": threshold
    })

    # split data
    left = {k: [] for k in data if k != feature}
    right = {k: [] for k in data if k != feature}

    for i in range(len(y)):
        target = left if data[feature][i] <= threshold else right
        for k in target:
            target[k].append(data[k][i])

    # recurse
    recursive_best_splits(left, splits, min_samples)
    recursive_best_splits(right, splits, min_samples)


def best_split(data):
    import math

    X_features = [k for k in data.keys() if k != "Buy"]
    y = data["Buy"]
    n = len(y)

    def gini(labels):
        if not labels:
            return 0
        p = sum(labels) / len(labels)
        return 1 - p**2 - (1 - p)**2

    best = {
        "feature": None,
        "threshold": None,
        "gini": float("inf"),
        "left_indices": None,
        "right_indices": None
    }

    for feature in X_features:
        values = data[feature]
        sorted_unique = sorted(set(values))

        # candidate thresholds = midpoints
        thresholds = [
            (sorted_unique[i] + sorted_unique[i + 1]) / 2
            for i in range(len(sorted_unique) - 1)
        ]

        for t in thresholds:
            left_idx = [i for i, v in enumerate(values) if v <= t]
            right_idx = [i for i, v in enumerate(values) if v > t]

            if not left_idx or not right_idx:
                continue

            g_left = gini([y[i] for i in left_idx])
            g_right = gini([y[i] for i in right_idx])

            weighted_gini = (
                len(left_idx) / n * g_left +
                len(right_idx) / n * g_right
            )

            if weighted_gini < best["gini"]:
                best.update({
                    "feature": feature,
                    "threshold": t,
                    "gini": weighted_gini,
                    "left_indices": left_idx,
                    "right_indices": right_idx
                })

    return best

splits = []
recursive_best_splits(data, splits)

plt.figure(figsize=(7, 6))

# scatter plot
plt.scatter(
    data["Income"],
    data["Age"],
    c=data["Buy"],
    cmap="bwr",
    edgecolors="k"
)

plt.xlabel("Income")
plt.ylabel("Age")
plt.title("Data with Decision Splits")


for s in splits:
    if s["feature"] == "Income":
        plt.axvline(
            x=s["threshold"],
            linestyle="--",
            linewidth=2
        )
    elif s["feature"] == "Age":
        plt.axhline(
            y=s["threshold"],
            linestyle="--",
            linewidth=2
        )


plt.show()