FROM python:3.11-slim

LABEL maintainer="Dev A"
LABEL description="WhatsApp Sales RL – OpenEnv server"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# COPY expects your repo to already have the package structure:
# env/, reward/, tasks/, ui/ as subdirectories
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]