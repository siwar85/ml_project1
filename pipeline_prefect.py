"""
pipeline_prefect.py
Orchestration du pipeline ML (Churn Modelling) avec Prefect.
Les tâches appellent main.py via ses étapes (--step).
"""

import argparse
import os
import shutil
import subprocess
import sys

from prefect import flow, task

from model_pipeline import load_model, prepare_data

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TEST_FILE = "test_model_pipeline.py"
REPO_URL = "https://github.com/siwar85/ml_project1.git"
CLONE_DIR = "projet_clone"


def run_command(command):
    """Exécute une commande système et lève une erreur en cas d'échec."""
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Échec de la commande : {' '.join(command)}")


def run_main(step):
    """Lance une étape de main.py."""
    run_command(
        [
            sys.executable,
            "main.py",
            "--step",
            step,
            "--data",
            DATA_PATH,
            "--model",
            MODEL_PATH,
        ]
    )


# ===== TASKS : GIT ET DÉPENDANCES =====


@task(name="Récupération du projet (git)")
def fetch_project():
    if os.path.isdir(".git"):
        run_command(["git", "pull", "--ff-only"])
    else:
        run_command(["git", "clone", REPO_URL, CLONE_DIR])


@task(name="Installer les dépendances")
def install_dependencies():
    run_command(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
    )


# ===== TASKS : QUALITÉ ET SUIVI DU CODE =====


@task(name="Formatage du code")
def format_code():
    run_command([sys.executable, "-m", "black", *PROJECT_FILES])


@task(name="Qualité du code")
def check_quality():
    run_command(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *PROJECT_FILES]
    )


@task(name="Sécurité du code")
def check_security():
    run_command([sys.executable, "-m", "bandit", "-ll", "-q", *PROJECT_FILES])


@task(name="Tests unitaires")
def run_unit_tests():
    run_command([sys.executable, "-m", "pytest", "-q", TEST_FILE])


# ===== TASKS : DONNÉES ET MODÈLE (via main.py) =====


@task(name="Préparation des données")
def prepare_data_task():
    run_main("prepare")


@task(name="Entraînement et sauvegarde du modèle")
def train_model_task():
    run_main("train")


@task(name="Chargement du modèle")
def load_model_task():
    run_main("load")


@task(name="Évaluation du modèle")
def evaluate_model_task():
    run_main("evaluate")


@task(name="Prédiction")
def predict_task():
    _, x_test, _, _ = prepare_data(DATA_PATH)
    model = load_model(MODEL_PATH)
    echantillon = x_test.head(5)
    predictions = model.predict(echantillon)
    for index, prediction in zip(echantillon.index, predictions):
        result = "va quitter la banque" if prediction == 1 else "reste client"
        print(f"Client {index} : {result}")


# ===== TASK : NETTOYAGE =====


@task(name="Nettoyage des fichiers générés")
def clean_files():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"Supprimé : {MODEL_PATH}")
    for folder in ["__pycache__", ".pytest_cache"]:
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
            print(f"Supprimé : {folder}")


# ===== FLOWS =====


@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="test")
def test_flow():
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="code")
def code_flow():
    format_code()
    test_flow()


@flow(name="prepare")
def prepare_flow():
    prepare_data_task()


@flow(name="train")
def train_flow():
    prepare_data_task()
    train_model_task()


@flow(name="evaluate")
def evaluate_flow():
    load_model_task()
    evaluate_model_task()


@flow(name="predict")
def predict_flow():
    predict_task()


@flow(name="clean")
def clean_flow():
    clean_files()


@flow(name="all")
def all_flow():
    fetch_project()
    code_flow()
    prepare_data_task()
    train_model_task()
    evaluate_model_task()


FLOWS = {
    "install": install_flow,
    "test": test_flow,
    "code": code_flow,
    "prepare": prepare_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "clean": clean_flow,
    "all": all_flow,
}


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument(
        "--flow", choices=FLOWS.keys(), required=True, help="Flow à lancer"
    )
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
