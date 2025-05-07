# Базовый образ
FROM python:3.12.5

# Рабочая директория внутри контейнера
WORKDIR /usr/src/app

# Копируем код
COPY ./src ./src
COPY ./requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт 8000
EXPOSE 8000

# Команда запуска FastAPI через Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--app-dir", "src"]
