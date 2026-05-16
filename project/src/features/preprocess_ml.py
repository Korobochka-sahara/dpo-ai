from pathlib import Path
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from configs.config import PROCESSED_DIR

nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

DATA_DIR = PROCESSED_DIR

FILES = ["train.csv", "test.csv"]


def clean_text(text: str) -> str:

    tokens = text.split()

    # stopwords + lemmatization
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOPWORDS and len(word) > 2
    ]

    return " ".join(tokens)


for file_name in FILES:

    df = pd.read_csv(DATA_DIR / file_name)

    df["processed_text"] = df["text"].astype(str).apply(clean_text)

    output_name = file_name.replace(".csv", "_ml.csv")

    df.to_csv(DATA_DIR / output_name, index=False)

    print(f"Saved: {output_name}")