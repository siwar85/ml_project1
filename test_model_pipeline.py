"""Tests unitaires de model_pipeline.py."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model_pipeline import load_model, prepare_data, save_model, train_model


def make_sample_csv(path, n_rows=50):
    """Crée un petit jeu de données factice au format Churn_Modelling."""
    data = pd.DataFrame(
        {
            "RowNumber": range(1, n_rows + 1),
            "CustomerId": range(15000000, 15000000 + n_rows),
            "Surname": ["Test"] * n_rows,
            "CreditScore": [600 + i for i in range(n_rows)],
            "Geography": ["France"] * n_rows,
            "Gender": ["Female", "Male"] * (n_rows // 2),
            "Age": [30 + i % 30 for i in range(n_rows)],
            "Tenure": [i % 10 for i in range(n_rows)],
            "Balance": [1000.0 * i for i in range(n_rows)],
            "NumOfProducts": [1 + i % 3 for i in range(n_rows)],
            "HasCrCard": [i % 2 for i in range(n_rows)],
            "IsActiveMember": [(i + 1) % 2 for i in range(n_rows)],
            "EstimatedSalary": [50000.0 + 100 * i for i in range(n_rows)],
            "Exited": [i % 2 for i in range(n_rows)],
        }
    )
    data.to_csv(path, index=False)
    return str(path)


def test_prepare_data_shapes(tmp_path):
    csv_path = make_sample_csv(tmp_path / "sample.csv")
    x_train, x_test, y_train, y_test = prepare_data(csv_path)
    assert x_train.shape == (40, 9)
    assert x_test.shape == (10, 9)
    assert len(y_train) == 40
    assert len(y_test) == 10
    assert "Exited" not in x_train.columns


def test_train_model_returns_random_forest(tmp_path):
    csv_path = make_sample_csv(tmp_path / "sample.csv")
    x_train, x_test, y_train, y_test = prepare_data(csv_path)
    model = train_model(x_train, y_train, n_estimators=10)
    assert isinstance(model, RandomForestClassifier)
    assert len(model.predict(x_test)) == len(x_test)


def test_save_and_load_model(tmp_path):
    csv_path = make_sample_csv(tmp_path / "sample.csv")
    x_train, x_test, y_train, y_test = prepare_data(csv_path)
    model = train_model(x_train, y_train, n_estimators=10)
    model_path = str(tmp_path / "model.joblib")
    save_model(model, model_path)
    loaded = load_model(model_path)
    assert (loaded.predict(x_test) == model.predict(x_test)).all()
