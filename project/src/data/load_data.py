#Загрузка данных с Kaggle bittlingmayer/amazonreviews
from pathlib import Path
import zipfile
import kaggle
from configs.config import DATA_DIR


RAW_DATA_DIR = Path(DATA_DIR / "raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():

    kaggle.api.dataset_download_files(
        "bittlingmayer/amazonreviews",
        path=RAW_DATA_DIR,
        unzip=False
    )

    zip_path = RAW_DATA_DIR / "amazonreviews.zip"

    print(f"Downloaded: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    print("Dataset extracted successfully")


if __name__ == "__main__":
    download_dataset()