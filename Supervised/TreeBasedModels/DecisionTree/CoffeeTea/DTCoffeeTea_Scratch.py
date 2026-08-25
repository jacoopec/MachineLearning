#The algorithm chooses the feature that minimizes Gini impurity at each split.
#Recursively builds a nested dictionary representing the tree.
#Each leaf contains a class label ('Tea' or 'Coffee').

import pandas as pd
from collections import Counter


def write_tree_to_file(tree, filename, indent=""):
    with open(filename, "w") as f:
        def recurse(node, level_indent):
            if isinstance(node, dict):
                for key, branch in node.items():
                    for value, subtree in branch.items():
                        f.write(f"{level_indent}{key} = {value}:\n")
                        recurse(subtree, level_indent + "  ")
            else:
                f.write(f"{level_indent}→ {node}\n")

        recurse(tree, indent)


# 1. Dataset (same as before)
data = {
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Rainy'],
    'Mood':    ['Happy', 'Tired', 'Happy', 'Tired', 'Tired', 'Happy', 'Happy', 'Tired'],
    'Drink':   ['Tea', 'Coffee', 'Tea', 'Coffee', 'Coffee', 'Tea', 'Tea', 'Coffee']
}
df = pd.DataFrame(data)

# 2. Gini Impurity
def gini_impurity(labels):
    total = len(labels)
    counts = Counter(labels)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

# 3. Split the dataset by a feature's value
def split_dataset(data, feature, value):
    return data[data[feature] == value]

# 4. Find the best feature to split
def find_best_split(data, features):
    best_gini = 1
    best_feature = None
    best_splits = None
    
    for feature in features:
        values = data[feature].unique()
        splits = [split_dataset(data, feature, v) for v in values]
        weighted_gini = sum((len(split) / len(data)) * gini_impurity(split['Drink']) for split in splits)
        
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_feature = feature
            best_splits = {v: split for v, split in zip(values, splits)}
    
    return best_feature, best_splits

# 5. Recursively build the tree
def build_tree(data, features, depth=0):
    labels = data['Drink']
    if len(set(labels)) == 1:
        return labels.iloc[0]
    if not features:
        return Counter(labels).most_common(1)[0][0]
    
    best_feature, best_splits = find_best_split(data, features)
    if best_feature is None:
        return Counter(labels).most_common(1)[0][0]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value, subset in best_splits.items():
        tree[best_feature][value] = build_tree(subset, remaining_features, depth + 1)

    return tree

# 6. Build and print the tree
features = ['Weather', 'Mood']
tree = build_tree(df, features)


# Build the tree
features = ['Weather', 'Mood']
tree = build_tree(df, features)

# Save to file
write_tree_to_file(tree, "decision_tree_output.txt")

print("✅ Decision tree saved to 'decision_tree_output.txt'")

