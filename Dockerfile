FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    MEDIA_DIR=/app/media

WORKDIR /app

COPY requirements.txt .

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        pkg-config \
        libfreetype6-dev \
        libpng-dev; \
    pip install --no-cache-dir -r requirements.txt; \
    apt-get purge -y --auto-remove build-essential libpq-dev pkg-config \
        libfreetype6-dev libpng-dev; \
    rm -rf /var/lib/apt/lists/*

COPY yiz_ai.py .

RUN set -eux; \
    useradd --uid 1000 --create-home --shell /usr/sbin/nologin yiz; \
    mkdir -p /app/media /app/plugins; \
    chown -R 1000:1000 /app

USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "yiz_ai:app", "--host", "0.0.0.0", "--port", "8000"]
