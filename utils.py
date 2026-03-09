
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#MUHAB W ZIBRA KOLO
def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    return plt


def plot_feature_importance(feature_importance, feature_names):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    return plt


def create_patient_features(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal):
    """Create a feature array for a single patient"""
    return [[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
             exang, oldpeak, slope, ca, thal]]
