"""Déploiements planifiés des flows Prefect."""

from prefect import serve
from prefect.schedules import Cron

from pipeline_prefect import (
    all_flow,
    clean_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    predict_flow,
    prepare_flow,
    test_flow,
    train_flow,
)

TZ = "Africa/Tunis"

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedule=Cron("0 2 * * *", timezone=TZ),
            tags=["full-pipeline", "mlops"],
        ),
        clean_flow.to_deployment(
            name="ml-pipeline-clean", tags=["maintenance", "mlops"]
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code", tags=["code-quality", "mlops"]
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        predict_flow.to_deployment(
            name="ml-pipeline-predict", tags=["prediction", "mlops"]
        ),
        prepare_flow.to_deployment(name="ml-pipeline-prepare", tags=["data", "mlops"]),
        test_flow.to_deployment(name="ml-pipeline-test", tags=["testing", "mlops"]),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
    )
