
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from knn import KNN


class HeartDiseaseModel:
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(max_iter=1000),
            'svm': SVC(probability=True),
            'decision_tree': DecisionTreeClassifier(),
            'knn': KNN(k=3)
        }
        self.best_model = None
        self.best_accuracy = 0

    def train_models(self, X_train, X_test, y_train, y_test):
        results = {}

        for name, model in self.models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model

        return results

    def predict(self, features):
        if self.best_model is None:
            raise Exception("Model needs to be trained first!")
        if hasattr(self.best_model, 'predict_proba'):
            proba = self.best_model.predict_proba(features)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1:2]
            return proba
        pred = self.best_model.predict(features)
        return np.array(pred, dtype=float).reshape(-1, 1)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.best_model = pickle.load(f)
