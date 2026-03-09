# Heart Disease Prediction System

A Streamlit web app that predicts heart disease risk using multiple machine learning models (Logistic Regression, SVM, Decision Tree, and a custom KNN). Includes feature selection, model comparison, and an interactive prediction form.

## Features

- **Data pipeline**: Load and clean heart disease CSV, handle missing values and outliers, encode categorical variables.
- **Feature selection**: Random Forest–based feature importance; configurable number of features (5–20) via sidebar.
- **Multiple models**: Trains and compares Logistic Regression, SVM, Decision Tree, and custom KNN; keeps the best by accuracy.
- **Evaluation**: Accuracy, classification report, and confusion matrix per model.
- **Interactive prediction**: Input patient features and get a heart-disease probability; same encoding and scaling as training.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies.

## Setup

```bash
# Clone the repo (after you've pushed to GitHub)
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction

# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Run the app

Place `heart.csv` in the project root (same folder as `main.py`), then:

```bash
streamlit run main.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Project structure

- `main.py` – Streamlit UI, training, and prediction.
- `model.py` – `HeartDiseaseModel`: trains multiple classifiers and uses the best for prediction (with probability support).
- `knn.py` – Custom KNN classifier (fit, predict, predict_proba).
- `data_preprocessing.py` – Load, clean, outlier detection, encoding, feature selection, train/test split.
- `utils.py` – Correlation plot and patient feature builder.
- `heart.csv` – Heart disease dataset (features + `target`).
- `requirements.txt` – Python dependencies.

## Dataset

Use a heart disease dataset with columns: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`. Missing values can be `?`; they are dropped. Numeric outliers are removed with IQR.

## License

MIT (or your choice).
