from pathlib import Path
import joblib

from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from configs.config import MODEL_DIR


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def export_artifacts(
    model,
    label_encoder,
    vectorizer=None,
    tokenizer=None
):

    # sklearn / catboost
    if not isinstance(model, PreTrainedModel):

        joblib.dump(
            model,
            MODEL_DIR / "model.pkl"
        )

    else:
        model.save_pretrained(
            MODEL_DIR / "bert_model"
        )

    # TF-IDF vectorizer
    if vectorizer is not None:

        joblib.dump(
            vectorizer,
            MODEL_DIR / "vectorizer.pkl"
        )

    # tokenizer
    if tokenizer is not None:

        tokenizer.save_pretrained(
            MODEL_DIR / "tokenizer"
        )

    # label encoder
    joblib.dump(
        label_encoder,
        MODEL_DIR / "label_encoder.pkl"
    )

    print("Artifacts exported successfully")