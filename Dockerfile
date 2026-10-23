FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app app
COPY core core
COPY infra infra

# Unprivilegierter Benutzer, /data ist für die SQLite-Datei mit dem Chatverlauf
RUN useradd --create-home --uid 10001 appuser \
    && mkdir /data \
    && chown appuser /data
USER appuser

EXPOSE 8501

# Standard: die Chat-Oberfläche. Der Sync-Dienst überschreibt das Kommando (siehe compose.yaml).
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
