FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt

COPY api.py src/ data/ config/ ./

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=2 CMD curl -f http://localhost:8001/health || exit 1
CMD ["python", "api.py"]

