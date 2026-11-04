# Imagem de treino: sem uvicorn/fastapi, com cliente MLflow e DVC. Contexto: raiz do repo.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY shared shared
COPY pipeline/pyproject.toml pipeline/pyproject.toml
COPY pipeline/src pipeline/src
RUN pip install --no-cache-dir -e shared -e "pipeline[dev]" boto3 "dvc[s3]==3.67.1"
COPY . .
WORKDIR /repo/pipeline
CMD ["dvc", "repro"]
