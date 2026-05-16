from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from configs.config import PROCESSED_DIR, MODEL_DIR

DATA_DIR = PROCESSED_DIR

train_df = pd.read_csv(DATA_DIR / "train_ml.csv")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

vectorizer.fit(train_df["processed_text"])

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print("TF-IDF vectorizer saved.")