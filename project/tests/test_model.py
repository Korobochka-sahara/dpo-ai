from src.models.inference import SentimentModel


def test_model_prediction():

    model = SentimentModel()

    result = model.predict(
        "Amazing quality product"
    )

    assert isinstance(result, dict)

    assert "prediction" in result

    assert "confidence" in result


def test_prediction_classes():

    model = SentimentModel()

    result = model.predict(
        "Terrible movie"
    )

    assert result["prediction"] in [
        "positive",
        "negative"
    ]


def test_confidence_range():

    model = SentimentModel()

    result = model.predict(
        "Good product"
    )

    assert 0 <= result["confidence"] <= 1