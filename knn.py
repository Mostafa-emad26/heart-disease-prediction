import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):

        distances = np.linalg.norm(self.X_train - x, axis=1)


        k_indices = np.argsort(distances)[:self.k]


        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _predict_proba_single(self, x):
        """Probability of class 1 = fraction of k nearest neighbors that are positive."""
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        return sum(k_nearest_labels) / len(k_nearest_labels)

    def predict_proba(self, X):
        """Return probability of class 1 (heart disease) for each sample."""
        probas = [self._predict_proba_single(x) for x in X]
        return np.array(probas).reshape(-1, 1)