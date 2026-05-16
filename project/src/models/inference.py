import joblib
import numpy as np
import torch

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from configs.config import MODEL_DIR


class SentimentModel:

    def __init__(self):

        self.label_encoder = joblib.load(
            MODEL_DIR / "label_encoder.pkl"
        )

        # =========================
        # BERT
        # =========================

        if (MODEL_DIR / "bert_model").exists():

            self.model_type = "bert"

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR / "tokenizer"
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_DIR / "bert_model"
            )

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self.model.to(self.device)

            self.model.eval()

        # =========================
        # sklearn / catboost
        # =========================

        else:

            self.model_type = "classic"

            self.model = joblib.load(
                MODEL_DIR / "model.pkl"
            )

            self.vectorizer = joblib.load(
                MODEL_DIR / "vectorizer.pkl"
            )

    def predict(self, text: str):

        text = text.lower().strip()
        
        # =========================
        # BERT
        # =========================

        if self.model_type == "bert":

            encoding = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(
                    outputs.logits,
                    dim=1
                )

                prediction = torch.argmax(
                    probs,
                    dim=1
                ).item()

                confidence = probs.max().item()

        # =========================
        # sklearn / catboost
        # =========================

        else:

            X = self.vectorizer.transform([text])

            prediction = self.model.predict(X)[0]

            probabilities = self.model.predict_proba(X)[0]

            confidence = float(np.max(probabilities))

        label = self.label_encoder.inverse_transform(
            [prediction]
        )[0]

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "model_type": self.model_type
        }