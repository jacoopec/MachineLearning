import numpy as np

class MultinomialNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Class priors
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Feature counts per class (for likelihoods)
        self._feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        self._class_counts = np.zeros(n_classes, dtype=np.float64)  # Total feature counts per class

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            self._feature_counts[idx, :] = X_c.sum(axis=0)
            self._class_counts[idx] = self._feature_counts[idx, :].sum()

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            # Log prior
            log_prior = np.log(self._priors[idx])

            # Log likelihood with Laplace smoothing (add-1)
            class_count = self._class_counts[idx]
            smoothed_fc = self._feature_counts[idx, :] + 1.0
            smoothed_cc = class_count + x.shape[0]  # + number of features
            log_likelihood = np.sum(x * np.log(smoothed_fc / smoothed_cc))

            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]