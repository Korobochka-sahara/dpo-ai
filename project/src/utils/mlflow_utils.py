import mlflow

from configs.config import MLFLOW_DIR


MLFLOW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def setup_mlflow(
    experiment_name="sentiment_analysis"
):

    tracking_uri = (
        f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').resolve()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    try:
        mlflow.end_run()
    except Exception:
        pass

    print("MLflow URI:", tracking_uri)

    return mlflow