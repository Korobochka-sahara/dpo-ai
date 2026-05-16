import mlflow
from contextlib import contextmanager

@contextmanager
def safe_run(run_name: str):

    # полностью сбрасываем состояние
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    run = None

    try:
        run = mlflow.start_run(run_name=run_name)
        yield run

    except Exception as e:
        print(f"[MLflow error handled]: {e}")

    finally:
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass