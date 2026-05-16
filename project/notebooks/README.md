# Ноутбуки проекта

В данной папке находятся Jupyter notebooks с экспериментами и анализом данных.

## Структура

### `eda.ipynb`

Разведочный анализ данных:
- распределение классов;
- анализ длины текстов;
- визуализация датасета;
- базовая статистика.

---

### `experiments/`

Эксперименты с моделями:

- `01_logreg_tfidf.ipynb`
  - Logistic Regression + TF-IDF;

- `02_NaiveBayes.ipynb`
  - Multinomial Naive Bayes;

- `03_CatBoost.ipynb`
  - CatBoost classifier;

- `04_DistilBert.ipynb`
  - DistilBERT fine-tuning через HuggingFace.

---

### `leaders.ipynb`

Сравнение моделей:
- accuracy;
- F1-score;
- leaderboard экспериментов MLflow.

---

## MLflow

Все эксперименты логируются через MLflow:
- параметры;
- метрики;
- модели;
- артефакты.