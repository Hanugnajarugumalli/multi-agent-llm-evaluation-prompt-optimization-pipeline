FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY billsum_v4_1/ ./billsum_v4_1/

# default is small run; change CMD for full eval
CMD ["python", "scripts/run_experiment.py", "--limit", "15", "--run-name", "docker_smoke"]
