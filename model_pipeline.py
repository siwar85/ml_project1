"""
model_pipeline.py
Fonctions modulaires du pipeline ML - Prédiction du churn bancaire.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(csv_path, test_size=0.2, random_state=1):
    """
    Charge et prétraite les données.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine pour la reproductibilité.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # 1. Chargement
    df = pd.read_csv(csv_path)

    # 2. Vérifications
    if df.isna().sum().sum() > 0:
        print("Valeurs manquantes détectées : suppression.")
        df = df.dropna()
    if df.duplicated().any():
        print("Doublons détectés : suppression.")
        df = df.drop_duplicates()

    # 3. Encodage de Gender
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # 4. Suppression des colonnes inutiles
    df = df.drop(columns=["Surname", "Geography", "RowNumber", "CustomerId"])

    # 5. Features / cible
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    # 6. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Données préparées : {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un modèle RandomForest.

    Args:
        X_train: features d'entraînement.
        y_train: cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine pour la reproductibilité.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.

    Args:
        model: modèle entraîné.
        X_test: features de test.
        y_test: cible de test.

    Returns:
        float: accuracy du modèle.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, path="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné avec joblib.

    Args:
        model: modèle à sauvegarder.
        path (str): chemin du fichier de sortie.
    """
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")


def load_model(path="classifier.joblib"):
    """
    Charge un modèle sauvegardé avec joblib.

    Args:
        path (str): chemin du fichier du modèle.

    Returns:
        modèle chargé.
    """
    model = joblib.load(path)
    print(f"Modèle chargé depuis : {path}")
    return model
