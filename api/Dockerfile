# Contexto de build: raiz do repo (docker compose build api).
FROM python:3.12-slim AS builder
WORKDIR /build
COPY api/pyproject.toml ./
COPY shared /build/shared
# Instala só as dependências (versões pinadas no pyproject) num prefixo copiável.
RUN python - <<'PY' > requirements.txt
import tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
print("\n".join(d for d in deps if not d.startswith("resonance-features")))  # vem de ./shared, não do PyPI
PY
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt ./shared

FROM python:3.12-slim
WORKDIR /srv/api
COPY --from=builder /install /usr/local
COPY api/app ./app
RUN useradd --system --uid 1001 api && mkdir -p /srv/data/prediction_logs && chown -R api /srv/data
USER api
ENV MODEL_REGISTRY_URI=models:/resonance@production \
    DATA_PATH=/srv/data/processed/tracks.csv \
    PREDICTION_LOG_DIR=/srv/data/prediction_logs
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/v1/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
