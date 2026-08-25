import numpy as np
import math

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1     # direction of inequality
        self.prediction_left = None
        self.prediction_right = None

    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples

        min_error = float("inf")

        # Try all features
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            # Try all thresholds
            for t in thresholds:
                # Predict: left side <= threshold, right side > threshold
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t

                # For each side, choose the majority class (weighted)
                left_pred = 1 if np.sum(sample_weights[left_idx] * (y[left_idx] == 1)) >= \
                                np.sum(sample_weights[left_idx] * (y[left_idx] == -1)) else -1

                right_pred = 1 if np.sum(sample_weights[right_idx] * (y[right_idx] == 1)) >= \
                                 np.sum(sample_weights[right_idx] * (y[right_idx] == -1)) else -1
                print(left_pred)
                print(right_pred)
                # Compute weighted error
                y_pred = np.zeros(n_samples)
                y_pred[left_idx] = left_pred
                y_pred[right_idx] = right_pred

                error = np.sum(sample_weights[y_pred != y])

                # Save the best stump
                if error < min_error:
                    min_error = error
                    self.feature_index = feature
                    self.threshold = t
                    self.prediction_left = left_pred
                    self.prediction_right = right_pred

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        left_idx = X[:, self.feature_index] <= self.threshold
        right_idx = X[:, self.feature_index] > self.threshold

        y_pred[left_idx] = self.prediction_left
        y_pred[right_idx] = self.prediction_right

        return y_pred

class DecisionStumpGini:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.prediction_left = None
        self.prediction_right = None
        self.amount_of_say = None
        self.n_features = None

    # -------------------------
    # Gini impurity calculation
    # -------------------------
    def gini(self, y):
        if len(y) == 0:
            return 0

        # count classes
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        # Gini formula: 1 - Σ p_i^2
        return 1 - np.sum(probs**2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features = X.shape[1]
        min_gini = float("inf")

        # Try every feature
        for feature in range(n_features-1):
            thresholds = np.unique(X[:, feature])

            # Try every threshold for that feature
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t

                y_left = y[left_idx]
                y_right = y[right_idx]

                # Compute Gini of both sides
                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)

                # Weighted gini:
                gini_total = (
                    len(y_left) / n_samples * gini_left +
                    len(y_right) / n_samples * gini_right
                )

                # Keep best split
                if gini_total < min_gini:
                    min_gini = gini_total
                    self.feature_index = feature
                    self.threshold = t

                    # majority class prediction
                    self.prediction_left = np.argmax(np.bincount(y_left)) if len(y_left) > 0 else 0
                    self.prediction_right = np.argmax(np.bincount(y_right)) if len(y_right) > 0 else 0
        
        self.predict(X)
        # self.amount_of_say = self._amount_of_say(X)


    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype = int)

        left_idx = X[:, self.feature_index] <= self.threshold
        right_idx = X[:, self.feature_index] > self.threshold

        y_pred[left_idx] = self.prediction_left
        y_pred[right_idx] = self.prediction_right

        preds = (y_pred == y)
        false_indexes = np.where(preds == False)[0]
        true_indexes = np.where(preds == True)[0]
        
        num_false = np.sum(preds == False)
        tot_err = num_false * X[0,self.n_features-1]
        self.amount_of_say = 0.5 * math.log((1-tot_err)/(tot_err))

        X[true_indexes,self.n_features-1] = math.exp(-self.amount_of_say) * X[true_indexes,self.n_features-1]
        X[false_indexes,self.n_features-1] = math.exp(self.amount_of_say) * X[false_indexes,self.n_features-1]
        
        error_sum = np.sum(X[:, self.n_features-1])
        X[:, self.n_features-1] = X[:, self.n_features-1] / error_sum


        return y_pred

# class Stump:
#     def __init__(self):
#         self.amount_of_say = None

# Example tiny dataset
# data = {
#     "feature3": [0, 4, 1, 2, 4, 5, 7, 8, 10, 5, 14, 20],   # NEW FEATURE
#     "feature2": [1, 1, 0, 2, 8, 9, 8, 10, 40, 30, 20, 30],
#     "feature1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
#     "class":    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
# }
data = {
    "feature3": [0, 4, 1, 2, 4, 5, 7, 8],   # NEW FEATURE
    "feature2": [1, 1, 0, 2, 8, 9, 8, 10],
    "feature1": [0, 0, 1, 0, 1, 1, 0, 1],
    "class":    [0, 0, 0, 0, 1, 1, 1, 1]
}

new_col = np.full((8),  1 / (8))

X = np.column_stack((
                        # data["feature3"],
                        # data["feature2"],
                        data["feature1"],
                        data["feature1"],
                        new_col
                    ))


y = np.array(data["class"])


stump = DecisionStumpGini()
stump.fit(X, y)

print("Feature:",          stump.feature_index)
print("Threshold:",        stump.threshold)
print("Left prediction:",  stump.prediction_left)
print("Right prediction:", stump.prediction_right)
print("Amount of say: ",   stump.amount_of_say)
print("Predictions:",      stump.predict(X))
print(X)
