from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MLFLOW_DIR = ARTIFACTS_DIR / "mlflow"

MODEL_DIR = ARTIFACTS_DIR / "models"

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"