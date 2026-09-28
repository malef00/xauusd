# Used by Hugging Face Spaces (or any Docker host)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DATA_SOURCE=twelvedata OPEN_BROWSER=0 PORT=7860
EXPOSE 7860
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 60
