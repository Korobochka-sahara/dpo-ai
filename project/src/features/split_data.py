from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from configs.config import PROCESSED_DIR

train = pd.read_csv(
    PROCESSED_DIR / "train.csv"
)

RANDOM_STATE = 42

DATA_PATH = Path(PROCESSED_DIR / "amazon_reviews_processed.csv")
OUTPUT_DIR = PROCESSED_DIR

df = pd.read_csv(DATA_PATH)

# train + test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["sentiment"],
    random_state=RANDOM_STATE
)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("Datasets saved:")
print(f"Train: {len(train_df)}")
print(f"Test: {len(test_df)}")