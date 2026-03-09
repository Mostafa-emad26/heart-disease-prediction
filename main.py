import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import load_and_clean_data, prepare_data, encode_categorical
from model import HeartDiseaseModel
from utils import plot_correlation_matrix, create_patient_features


def plot_feature_importance(feature_names, feature_scores, title="Feature Importance"):
    """Plot feature importance scores"""
    plt.figure(figsize=(10, 6))

    # Create a dictionary of feature names and their scores
    feature_dict = dict(zip(feature_names, feature_scores))

    # Sort features by importance
    sorted_features = sorted(feature_dict.items(), key=lambda x: x[1])
    features = [x[0] for x in sorted_features]
    scores = [x[1] for x in sorted_features]

    # Create the plot
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    return plt


def main():
    st.title('Heart Disease Prediction System')

    # Sidebar for feature selection options
    st.sidebar.header('Feature Selection Options')
    n_features = st.sidebar.slider('Number of Features to Select', 5, 20, 10)

    # Load and prepare data
    data = load_and_clean_data('heart.csv')
    X_train, X_test, y_train, y_test, scaler, selected_features, feature_scores, all_feature_names = prepare_data(
        data,
        n_features=n_features
    )

    # Display selected features
    st.header('Selected Features')
    st.write(f"Number of features selected: {len(selected_features)}")
    st.write("Selected features:", selected_features)

    # Plot feature importance
    st.header('Feature Importance')
    fig = plot_feature_importance(selected_features, feature_scores)
    st.pyplot(fig)

    # Create and train model
    model = HeartDiseaseModel()
    results = model.train_models(X_train, X_test, y_train, y_test)

    # Display model performance
    st.header('Model Performance')
    for model_name, result in results.items():
        st.subheader(f'{model_name.title()} Model')
        st.write(f'Accuracy: {result["accuracy"]:.2f}')
        st.text('Classification Report:')
        st.text(result['report'])

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name.title()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)
        plt.close()

    # Interactive Prediction
    st.header('Predict Heart Disease')

    # Create input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=20, max_value=100, value=50)
        sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', min_value=90, max_value=200, value=120)
        chol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    with col2:
        restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate', min_value=70, max_value=220, value=150)
        exang = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=6.0, value=0.0)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
        ca = st.selectbox('Number of Major Vessels', [0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

    if st.button('Predict'):
        # Create feature array for prediction
        features = create_patient_features(
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal
        )

        # Convert to DataFrame with feature names
        features_df = pd.DataFrame(features, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                                      'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                                                      'ca', 'thal'])

        # One-hot encode categorical variables
        categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        features_encoded = encode_categorical(features_df, categorical_columns, method='label')

        # Ensure all columns from training are present
        for col in all_feature_names:
            if col not in features_encoded.columns:
                features_encoded[col] = 0

        # Reorder columns to match training data
        features_encoded = features_encoded[all_feature_names]

        # Scale the features
        scaled_features = scaler.transform(features_encoded)

        # Convert selected feature names to indices
        selected_indices = [all_feature_names.index(f) for f in selected_features]
        scaled_features_selected = scaled_features[:, selected_indices]

        # Make prediction
        prediction_prob = model.predict(scaled_features_selected)

        # Display result
        st.subheader('Prediction Result')
        st.write(f'Probability of Heart Disease: {prediction_prob.item():.2%}')

        if prediction_prob > 0.5:
            st.error('High Risk of Heart Disease')
        else:
            st.success('Low Risk of Heart Disease')

    # Display correlation matrix
    st.header('Feature Correlations')
    fig = plot_correlation_matrix(data)
    st.pyplot(fig)


if __name__ == '__main__':
    main()
