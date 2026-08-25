import numpy as np

data = np.array([[1, 2], [2.5, 3], [3, 4], [2, 3], [3, 1],
                 [8, 5], [6, 6], [9, 6], [7, 8]])

labels = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1])

def augment_points(data, labels, n_aug=3, noise_std=0.2, seed=42):
    rng = np.random.default_rng(seed)

    augmented_data = [data]
    augmented_labels = [labels]

    for _ in range(n_aug):
        noise = rng.normal(loc=0, scale=noise_std, size=data.shape)
        new_points = data + noise

        augmented_data.append(new_points)
        augmented_labels.append(labels)

    return np.vstack(augmented_data), np.hstack(augmented_labels)


aug_data, aug_labels = augment_points(data, labels)

print(aug_data)
print(aug_labels)