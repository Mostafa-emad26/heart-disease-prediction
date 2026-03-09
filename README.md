# Heart Disease Prediction System

A web application that predicts whether a patient is at risk of heart disease based on clinical and demographic features. The system trains several machine learning classifiers, compares their performance, and uses the best-performing model to output a risk probability. The interface is built with Streamlit and allows users to explore the data, tune feature selection, and run predictions for new patients.

---

## What the Project Does

The project takes a heart disease dataset (e.g. UCI-style with attributes like age, blood pressure, cholesterol, ECG results, etc.) and:

1. **Cleans and prepares the data** so it is suitable for training.
2. **Selects a subset of the most important features** to improve model quality and reduce noise.
3. **Trains four different classifiers** on the same data and evaluates each one.
4. **Picks the best model** (by accuracy) and uses it to predict heart disease risk for new patient inputs.
5. **Exposes everything through a single Streamlit app**: feature importance, model comparison, and an interactive prediction form.

---

## Functionalities

### Data Loading and Cleaning

- **Load CSV**: Reads the heart disease dataset from a CSV file (`heart.csv`).
- **Handle missing values**: Replaces placeholder values (e.g. `?`) with missing and drops incomplete rows. Converts all columns to numeric and drops any remaining invalid rows.
- **Outlier removal**: For numeric columns (age, resting blood pressure, cholesterol, max heart rate, ST depression), outliers are detected using the **IQR (interquartile range)** method. Points below \(Q_1 - 1.5 \times \text{IQR}\) or above \(Q_3 + 1.5 \times \text{IQR}\) are removed so extreme values do not distort the models.
- **Categorical encoding**: Categorical variables (sex, chest pain type, fasting blood sugar, resting ECG, exercise-induced angina, slope, number of major vessels, thalassemia) are **label-encoded** so every column is numeric for the classifiers.

### Feature Selection

- **Importance-based selection**: A **Random Forest** classifier is trained on all features to compute **feature importance** scores.
- **Configurable count**: The user can choose how many top features to keep (between 5 and 20) via a sidebar slider. Only these features are used for training and prediction, which can improve generalization and speed.
- **Same features everywhere**: The same selected features and their order are used in training, evaluation, and when predicting for a new patient.

### Model Training and Comparison

- **Four classifiers** are trained on the same preprocessed data:
  - **Logistic Regression** – linear model that outputs a probability.
  - **Support Vector Machine (SVM)** – with probability estimates enabled.
  - **Decision Tree** – non-linear, interpretable rules.
  - **Custom K-Nearest Neighbors (KNN)** – implemented from scratch; uses Euclidean distance and majority vote among the \(k\) nearest training points; also provides a probability as the fraction of positive neighbors.
- **Best model selection**: After training, **accuracy** on the test set is computed for each model. The one with the **highest accuracy** is stored and used for all subsequent predictions.
- **Evaluation metrics**: For each model the app shows accuracy, a **classification report** (precision, recall, F1, support), and a **confusion matrix** so users can compare performance in detail.

### Prediction for New Patients

- **Interactive form**: The user enters one patient’s values for all input features (age, sex, chest pain type, blood pressure, cholesterol, etc.).
- **Same pipeline as training**: The input is label-encoded (to match training), scaled with the **same scaler** fit on the training data, and then only the **selected features** are passed to the best model.
- **Probability output**: The best model’s **probability of heart disease** (class 1) is shown. The UI labels the result as high or low risk based on a 0.5 threshold.

### Visualization and Exploration

- **Feature importance plot**: A bar chart of the selected features and their importance scores (from the Random Forest used for selection).
- **Correlation matrix**: A heatmap of correlations between all features in the dataset to help understand relationships in the data.

---

## How the Pieces Work Together

- **`data_preprocessing.py`** – Defines loading, cleaning, outlier detection, encoding, feature selection (Random Forest), scaling, and train/test split. The app uses these functions to build the same feature matrix for training and for a single patient.
- **`model.py`** – Defines `HeartDiseaseModel`: it holds the four classifiers, runs training and evaluation, selects the best by accuracy, and exposes a single `predict()` that returns heart-disease probability (using `predict_proba` when available, including for the custom KNN).
- **`knn.py`** – Custom KNN implementation: stores training data, computes Euclidean distances, finds the \(k\) nearest neighbors, and returns both the majority class and the proportion of positive neighbors as probability.
- **`utils.py`** – Helper to plot the correlation matrix and to build the feature vector for one patient from the form inputs.
- **`main.py`** – Streamlit app: calls the preprocessing and model code, displays feature selection, importance plot, model comparison (accuracy, report, confusion matrix), prediction form, and correlation heatmap.

---

## Dataset

The system expects a heart disease CSV with columns such as: **age**, **sex**, **cp** (chest pain type), **trestbps** (resting blood pressure), **chol** (cholesterol), **fbs** (fasting blood sugar), **restecg** (resting ECG), **thalach** (max heart rate), **exang** (exercise-induced angina), **oldpeak** (ST depression), **slope**, **ca** (number of major vessels), **thal** (thalassemia), and **target** (heart disease present or not). Missing values and outliers are handled as described above so the pipeline is robust to real-world noise.

---

## Summary

The Heart Disease Prediction System is a full pipeline: from raw CSV and optional tuning of feature count, to training and comparing multiple classifiers, to selecting the best model and using it to predict heart disease risk for new patients, with clear evaluation and visualizations throughout.
