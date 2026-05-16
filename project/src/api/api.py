from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.models.inference import SentimentModel


app = FastAPI(
    title="Sentiment Analysis API",
    description="""
REST API сервис для анализа тональности текстовых отзывов.

Функции:
- классификация текста;
- определение тональности;
- confidence score.

Поддерживаемые классы:
- positive
- negative

## Endpoints:
- `/predict` → Предсказание тональности для одного отзыва
- `/predict_batch` → Предсказание тональности для группы отзывов
- `/health` → сервесный health check

""",
    version="1.0.0"
)


model = SentimentModel()


class TextRequest(BaseModel):

        text: str = Field(
        ...,
        min_length=1,
        example="This product is amazing",
        description="Вставьте текст отзыва для анализа тональности"
    )


class PredictionResponse(BaseModel):

    prediction: str = Field(
        ...,
        example="positive",
        description="Предсказание тональности"
    )

    confidence: float = Field(
        ...,
        example=0.98,
        description="Точность предсказания тональности"
    )

class BatchTextRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_items=1,
        description="Вставьте несколько текстов для анализа тональности",
        example=[
            "This product is amazing",
            "Worst experience ever"
        ]
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


@app.get(
    "/",
    summary="Service status"
)
def root():

    """
    Проверка работы API.
    """

    return {
        "service": "sentiment-analysis",
        "status": "running"
    }


@app.get(
    "/health",
    summary="Health check"
)
def health():

    """
    Endpoint для проверки состояния сервиса.
    Используется для мониторинга и Docker health checks.
    """

    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Предсказание тональности"
)
def predict(request: TextRequest):

    """
    Анализирует текст и возвращает:
    - предсказание тональности;
    - точночть.
    """

    result = model.predict(request.text)

    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    summary="Несколько предсказаний тональности" 
)
def predict_batch(request: BatchTextRequest):

    results = []

    for text in request.texts:
        result = model.predict(text)

        results.append(
            PredictionResponse(
                prediction=result["prediction"],
                confidence=result["confidence"]
            )
        )

    return BatchPredictionResponse(predictions=results)