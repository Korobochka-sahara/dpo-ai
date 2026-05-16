FROM python:3.11-slim

WORKDIR /app

# системные зависимости (иногда нужны для torch/mlflow)
RUN apt-get update && apt-get install -y build-essential

# копируем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# копируем весь проект
COPY . .

# порт FastAPI
EXPOSE 8000

# запуск сервера
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]