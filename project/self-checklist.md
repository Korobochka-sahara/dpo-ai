# Самопроверка проекта (self-checklist)

Отметьте, что реализовано в проекте анализа тональности отзывов клиентов.

Проект включает:
- preprocessing текстовых данных;
- обучение и сравнение нескольких ML/DL моделей;
- MLflow tracking;
- FastAPI inference service;
- тестирование и воспроизводимость пайплайна.

В столбце «Да/Нет (студент)» поставьте:
- `✅` — реализовано;
- `❌` — отсутствует или не завершено.

В столбце «Где смотреть / комментарий» укажите соответствующие файлы проекта.

---

# Таблица самопроверки

| #  | Критерий | Да/Нет (студент) | Где смотреть / комментарий |
|----|-----------|------------------|-----------------------------|
| 1 | FastAPI сервис запускается по инструкции из README | ✅ | `README.md`, `src/API/api.py` |
| 2 | Endpoint `/predict` использует реальную обученную модель | ✅ | `src/models/inference.py`, `artifacts/models/` |
| 3 | Реализован health-check endpoint `/health` | ✅ | `src/API/api.py` |
| 4 | Есть EDA и визуализация данных | ✅ | `notebooks/eda.ipynb`, `artifacts/*.png` |
| 5 | Реализован preprocessing текстов | ✅ | `src/data/basic_data_clearing.py`, `src/features/preprocess_ml.py` |
| 6 | Выполнено разбиение train/test | ✅ | `src/features/split_data.py` |
| 7 | Реализована TF-IDF vectorization | ✅ | `src/features/vectorize_tfidf.py` |
| 8 | Есть baseline ML модель | ✅ | `notebooks/experiments/01_logreg_tfidf.ipynb` |
| 9 | Реализовано сравнение нескольких моделей | ✅ | `leaders.ipynb`, MLflow |
| 10 | Используется MLflow для логирования экспериментов | ✅ | `src/utils/mlflow_utils.py`, `artifacts/mlflow/` |
| 11 | Реализована нейросетевая модель DistilBERT | ✅ | `04_DistilBert.ipynb` |
| 12 | Выполнен подбор гиперпараметров | ✅ | notebooks с экспериментами |
| 13 | Сохраняются production artifacts | ✅ | `src/models/export_best_model.py`, `artifacts/models/` |
| 14 | Реализован inference pipeline | ✅ | `src/models/inference.py` |
| 15 | Код разделён по модулям и папкам | ✅ | `src/data`, `src/features`, `src/models`, `src/API` |
| 16 | Есть конфигурационный файл проекта | ✅ | `configs/config.py` |
| 17 | Реализованы pytest тесты | ✅ | `tests/` |
| 18 | Есть тесты API | ✅ | `tests/test_api.py` |
| 19 | Есть sanity-check тесты данных | ✅ | `tests/test_data.py` |
| 20 | В проекте отсутствуют реальные секреты и токены | ✅ | `.gitignore`, отсутствие `.env` |
| 21 | Есть sample dataset для демонстрации и тестов | ✅ | `data/sample/` |
| 22 | README описывает запуск и структуру проекта | ✅ | `README.md` |
| 23 | report.md содержит описание экспериментов и результатов | ✅ | `report.md` |
| 24 | Dockerfile для контейнеризации проекта | ✅ | `docker/Dockerfile` |

---
# Итоговая самооценка

## Реализовано в проекте

В рамках проекта реализованы:

- полный ML pipeline для анализа тональности текста;
- загрузка и preprocessing данных;
- очистка и нормализация текстов;
- TF-IDF векторизация;
- обучение и сравнение нескольких моделей машинного обучения;
- обучение transformer-модели DistilBERT;
- логирование экспериментов и метрик через MLflow;
- сохранение и экспорт production-модели;
- inference pipeline для получения предсказаний;
- REST API сервис на FastAPI;
- Swagger/OpenAPI документация;
- тестирование с использованием pytest;
- конфигурация проекта через отдельный config-модуль;
- Docker-сборка сервиса;
- EDA и визуализация результатов;
- unit и integration tests для основных компонентов проекта.

---

## Что можно улучшить в дальнейшем

В дальнейшем проект можно расширить следующими возможностями:

- более развитая система logging и monitoring;
- автоматический подбор гиперпараметров;
- хранение истории запросов пользователей;
- авторизация пользователей;
- поддержка multiclass sentiment analysis (`positive` / `neutral` / `negative`);
- автоматическое сохранение лучшей модели по результатам экспериментов.