# Исходный код проекта

В данной папке расположен основной код проекта.

## Структура

### `API/`

FastAPI сервис.

Содержит:
- endpoints;
- request/response schemas;
- запуск inference.

Основной endpoint:
- `/predict`

---

### `data/`

Загрузка и очистка данных:
- чтение raw dataset;
- preprocessing;
- подготовка CSV.

---

### `features/`

Подготовка признаков:
- TF-IDF vectorization;
- train/test split;
- preprocessing pipeline.

---

### `models/`

Inference и export моделей:
- загрузка production модели;
- prediction pipeline;
- export artifacts.

---

### `utils/`

Вспомогательные модули:
- MLflow setup;
- safe logging;
- utility functions.

---

## Используемые технологии

- FastAPI
- scikit-learn
- CatBoost
- HuggingFace Transformers
- PyTorch
- MLflow
- pytest