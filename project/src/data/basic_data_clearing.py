# Очистка данных
import bz2
import re
from pathlib import Path
from configs.config import PROCESSED_DIR, DATA_DIR

import pandas as pd


RAW_DATA_PATH = DATA_DIR / "raw/train.ft.txt.bz2"

""" SAMPLE_DIR = Path(DATA_DIR / "sample")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True) """


def clean_text(text: str) -> str:
    """
    Очистка текста
    """

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_line(line: str):
    """
    Парсинг строки отзыва.

    Пример:
    __label__2 This product is amazing
    """

    if line.startswith("__label__1"):
        sentiment = "negative"
        text = line.replace("__label__1", "").strip()

    else:
        sentiment = "positive"
        text = line.replace("__label__2", "").strip()

    return text, sentiment


def preprocess_dataset(max_samples=100000):

    texts = []
    sentiments = []
    seen_texts = set()
    duplicates_count = 0

    with bz2.open(RAW_DATA_PATH, "rt", encoding="utf-8") as file:

        for idx, line in enumerate(file):

            if idx >= max_samples:
                break

            text, sentiment = parse_line(line)

            text = clean_text(text)

            if len(text) < 5:
                continue

            if text in seen_texts:
                duplicates_count += 1
                continue

            seen_texts.add(text)
            texts.append(text)
            sentiments.append(sentiment)

            if idx % 10000 == 0:
                print(f"Processed {idx} reviews, Duplicates found: {duplicates_count}")

    df = pd.DataFrame({
        "text": texts,
        "sentiment": sentiments
    })

    print(df.head())

    print(df["sentiment"].value_counts())

    processed_path = (
        PROCESSED_DIR / "amazon_reviews_processed.csv"
    )

    df.to_csv(processed_path, index=False)

    print(f"Saved processed dataset to {processed_path}")

    """ sample_df = df.sample(
        n=1000,
        random_state=42
    )

    sample_path = (
        SAMPLE_DIR / "sample_reviews.csv"
    )

    sample_df.to_csv(sample_path, index=False)

    print(f"Saved sample dataset to {sample_path}") """


if __name__ == "__main__":
    preprocess_dataset()