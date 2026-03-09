import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def load_and_clean_data(file_path):

    data = pd.read_csv(file_path)

    data = data.replace('?', np.nan)
    data = data.dropna()

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna()
    data = data.reset_index(drop=True)

    numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    outliers = detect_outliers(data, numeric_columns, method='iqr')
    data = data[~outliers]
    data = data.reset_index(drop=True)

    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    data = encode_categorical(data, categorical_columns, method='label')

    return data


def detect_outliers(data, columns, method='iqr'):

    Q1 = data[columns].quantile(0.25)#bygeb elfirst quartile
    Q3 = data[columns].quantile(0.75)#bygeb elthird quartile
    IQR = Q3 - Q1#by7sb el iqr ely hya third minus elfirst
    outliers = ((data[columns] < (Q1 - 1.5 * IQR)) |#da range eloutlier zy ma khdna fe elstat
                (data[columns] > (Q3 + 1.5 * IQR))).any(axis=1)#Bygeb eloutliers ely homa elhgat ely bra elrange da
    return pd.Series(outliers, index=data.index)#byreturn boolean(true or false) value kol row outliier wla la


def encode_categorical(data, categorical_columns, method='label'):

    data_encoded = data.copy()


    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data_encoded[col] = label_encoder.fit_transform(data[col])

    return data_encoded


def prepare_data(data, n_features=10):


    X = data.drop('target', axis=1)
    y = data['target']


    all_feature_names = X.columns.tolist()


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    selected_features, feature_scores = select_features(X_scaled, y, n_features)
    X_selected = X_scaled[:, selected_features]
    selected_feature_names = X.columns[selected_features].tolist()


    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, selected_feature_names, feature_scores, all_feature_names


def select_features(X, y, n_features=10):

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_scores = rf.feature_importances_
    selected_features = np.argsort(feature_scores)[-n_features:]

    return selected_features, feature_scores