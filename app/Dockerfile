FROM python:3.10-slim

WORKDIR /app

# Копируем все файлы в контейнер
COPY . .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install pyTelegramBotAPI

# Переменная окружения для токена
ENV TELEGRAM_TOKEN=""

# Запускаем бота
CMD ["python", "telegram_bot.py"]
