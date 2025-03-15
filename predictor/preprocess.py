import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from predictor.Metadata import getmetadata


def load_data(file_path):
    """Loads and preprocesses dataset."""
    df = pd.read_csv(file_path)
    df = df.drop(["beats"], axis=1)

    df["class_name"] = df["class_name"].astype("category")
    df["class_label"] = df["class_name"].cat.codes
    lookup_genre_name = dict(zip(df.class_label.unique(), df.class_name.unique()))

    return df, lookup_genre_name


def split_data(df):
    """Splits data into training and testing sets."""
    X = df.iloc[:, 1:28]
    y = df["class_label"]
    return train_test_split(X, y, test_size=0.2, random_state=3)


def normalize_data(X_train, X_test):
    """Applies Min-Max Scaling to data."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def feature_importance(X_train_scaled, y_train, X):
    """Displays feature importance using Random Forest and Decision Tree."""
    models = {
        "Random Forest": RandomForestClassifier(random_state=0, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X.columns.values[i] for i in indices]

        plt.figure(figsize=(10, 5))
        plt.title(f"Feature Importance ({name})")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), names, rotation=90)
        plt.show()


def train_knn(X_train_scaled, y_train):
    """Trains a K-Nearest Neighbors (KNN) model."""
    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train_scaled, y_train)
    return knn


def train_svm(X_train_scaled, y_train):
    """Trains a Support Vector Machine (SVM) model."""
    svm = SVC(kernel="linear", C=10)
    svm.fit(X_train_scaled, y_train)
    return svm


def predict_genre(model, scaler, metadata, lookup_genre_name):
    """Predicts genre using the trained model."""
    metadata_array = np.array(metadata).reshape(1, -1)
    data_scaled = scaler.transform(metadata_array)
    prediction = model.predict(data_scaled)
    return lookup_genre_name[prediction[0]]


def save_models(scaler, knn, svm, lookup_genre_name):
    """Saves the trained models and necessary components."""
    models = {
        "scaler": scaler,
        "knn": knn,
        "svm": svm,
        "lookup_genre": lookup_genre_name,
    }
    with open("models.p", "wb") as f:
        pickle.dump(models, f)


def main(audiofile):
    # Load and preprocess data
    df, lookup_genre_name = load_data("data.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Normalize data
    scaler, X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

    # Feature importance visualization
    feature_importance(X_train_scaled, y_train, X_train)

    # Train models
    knn = train_knn(X_train_scaled, y_train)
    svm = train_svm(X_train_scaled, y_train)

    # Load metadata for prediction
    metadata = getmetadata(audiofile)

    # Predict genres using both models
    knn_prediction = predict_genre(knn, scaler, metadata, lookup_genre_name)
    svm_prediction = predict_genre(svm, scaler, metadata, lookup_genre_name)

    # Save models
    save_models(scaler, knn, svm, lookup_genre_name)

    return {"KNN": knn_prediction, "SVM": svm_prediction}


if __name__ == "__main__":
    audiofile = "aud.blues.wav"  # Example audio file
    predictions = main(audiofile)
    print(predictions)
