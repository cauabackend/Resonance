<div align="center">

# Resonance

**Ranking de demos musicais por potencial de popularidade dentro do gênero.**
De 500 demos, separa as ~50 que valem uma escuta atenta.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React_19-TypeScript-61DAFB?logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-model-EC6B23)
![MLflow](https://img.shields.io/badge/MLflow-registry-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-pipeline-945DD6?logo=dvc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
[![CI](https://github.com/cauabackend/Resonance/actions/workflows/ci.yml/badge.svg)](https://github.com/cauabackend/Resonance/actions/workflows/ci.yml)

[Resultados](#resultados) · [Como funciona](#como-funciona) · [Rodar](#como-rodar) · [Decisões (ADRs)](docs/adr) · [Limitações](#limitações-e-honestidade-dos-números)

</div>

---

## Resumo

Uma gravadora independente não tem orçamento para ouvir toda demo que recebe. O Resonance **ordena** as faixas por percentil de popularidade esperado *dentro do gênero* e explica cada nota, para que o A&R gaste a escuta onde ela rende mais.

| | |
|---|---|
| **Problema** | Priorizar demos com pouco tempo e dinheiro (ranking, não classificação binária) |
| **Dados** | 81.060 faixas do Spotify, 113 gêneros, 17.629 artistas |
| **Modelo** | XGBoost em duas modalidades: *com histórico do artista* e *só áudio* |
| **Resultado** | Num lote de 500 faixas, o top-50 do modelo tem percentil real médio de **76,1** (acaso: 50) |
| **Entrega** | API FastAPI + interface React + pipeline reproduzível (DVC/MLflow) + monitoramento de drift |
| **Rigor** | Splits e CV agrupados por artista, baselines, gate de promoção, 8 ADRs, testes e CI |

## Resultados

Todos os números vêm de **holdout e CV agrupados por artista** (o mesmo artista nunca está em treino e teste) e de `reports/evaluation.json`. NDCG@50 é medido em lotes aleatórios de 500 faixas: "o top-50 do modelo coincide com o top-50 real?".

| Holdout por artista | Spearman | NDCG@50 | ROC-AUC do badge | Baseline (Spearman / NDCG@50) |
|---|---|---|---|---|
| **Com histórico do artista** | **0,469** | **0,803** | **0,884** | média do artista: 0,301 / 0,733 |
| Só áudio (artista novo) | 0,144 | 0,625 | 0,800 | regressão linear: −0,003 / 0,551 |
| Nested CV, com histórico | 0,491 ± 0,003 | 0,828 ± 0,009 | | |
| Nested CV, só áudio | 0,136 ± 0,009 | 0,623 ± 0,014 | | |

- **O histórico do artista triplica o ranking** (Spearman 0,14 → 0,47) e abre a escala dos scores: desvio de 14,1 (p1 = 22, p99 = 84) contra 5,6 só com áudio.
- **Badge "é hit"** calibrado (isotônica, corte de custo 3:1): precisão 0,51 / recall 0,66 com histórico; ECE ≤ 0,014; Brier 0,062 contra 0,094 da taxa-base.
- **O artista domina a explicação:** `artist_context` pesa 9,8 pontos de percentil no SHAP, contra 1,5 do gênero e ≤ 1,0 de qualquer feature de áudio (estável entre folds: correlação de ranks 0,96).
- **Só áudio é fraco, e isso é o teto do tipo de feature**, não um bug. A literatura de *Hit Song Science* chega à mesma conclusão (Pachet & Roy, ISMIR 2008; Herremans et al., 2014).
- Leaderboard (NDCG@50, com histórico / só áudio): XGBRegressor 0,811 / 0,635 · Random Forest 0,809 / 0,632 · HistGradientBoosting 0,806 / 0,627 · Ridge 0,786 / 0,547 · XGBRanker 0,785 / 0,572 ([ADR 0002](docs/adr/0002-xgboost-regressor-e-nested-cv.md)).

### Desempenho da API

`python api/scripts/benchmark.py` (CPU, 1 worker uvicorn). Orçamento: p95 < 200 ms, checado no CI ([docs/performance-budget.md](docs/performance-budget.md)).

| Endpoint | Resultado |
|---|---|
| `POST /v1/predict` (sequencial, com SHAP) | p50 20,6 ms · p95 22,2 ms |
| `POST /v1/predict`, 8 clientes simultâneos | p95 173 ms · 57 req/s |
| `POST /v1/predict/batch` (500 faixas, ordenado) | 36 ms · ~13.800 faixas/s |

## Como funciona

Três sistemas desacoplados no mesmo monorepo, mais um pacote de features compartilhado:

```
web/       React 19 + TypeScript + Bootstrap (Sass), tema escuro monocromático, gráficos Plotly
api/       FastAPI: /v1/predict, /v1/predict/batch (ordenado), /v1/tracks/*, /v1/model/metadata, /v1/health[/deep]
pipeline/  DVC + MLflow: validate → preprocess → target → analyze → train → evaluate (gate) → registry
shared/    resonance_features: a MESMA engenharia de features no treino e na API (sem train/serve skew)
```

```
React ──/v1/predict──▶ FastAPI ──carrega artefato──▶ MLflow Model Registry (artefatos no MinIO)
                          │ logs de predição (JSONL)             ▲ promove se passar o gate
                          ▼                                      │
          Evidently (drift de features + de percentile_score) ◀── pipeline (DVC)
```

**Decisões que valem destacar**

- **Ranking por gênero.** O alvo é o percentil de popularidade dentro do gênero, então "90" em grindcore vale o mesmo que em pop ([ADR 0006](docs/adr/0006-ranking-em-vez-de-classificacao-binaria.md)).
- **Duas modalidades no mesmo modelo.** Com histórico (artista já em catálogo) e só áudio (demo de artista novo); o treino mascara o histórico em 50% das linhas e o XGBoost trata o NaN nativamente ([ADR 0008](docs/adr/0008-historico-do-artista-e-duas-modalidades.md)).
- **Sem vazamento.** Splits agrupados pelo artista principal (colaborações "A;B" contam como A) e histórico *leave-one-out*.
- **Gate de promoção.** O modelo só vai para `production` se vencer os baselines nas duas modalidades; senão o `dvc repro` falha.
- **Transparência por predição.** Cada resposta traz `percentile_score`, popularidade esperada, probabilidade calibrada do badge, limiar, versão do modelo e SHAP agrupado por fator ([ADR 0007](docs/adr/0007-isotonica-em-vez-de-platt.md)).
- **MLOps sem nuvem paga.** DVC, MLflow, MinIO e Evidently rodam localmente ([ADR 0005](docs/adr/0005-mlops-sem-nuvem-paga.md)).
- **Segurança operacional.** Checagem de SHA256 do artefato, `/v1/health/deep`, rate limit e logs estruturados.

### Telas

- **Busca:** escolhe uma faixa e vê gauge, radar, comparação com o gênero, simulador **"E se…"** (sliders que recalculam o score na API) e o porquê da decisão.
- **Ranking de demos:** sobe um CSV ([exemplo](web/public/exemplo-demos.csv)), recebe o lote ordenado com `rank`, modalidade e corte do top-50, e exporta em CSV.
- **Model card:** métricas por modalidade e limites de uso. Há também um toggle de alto contraste.

## Como rodar

### Docker

```bash
cp .env.example .env                              # defina MINIO_ROOT_PASSWORD (obrigatório)
docker compose up --build                         # minio + mlflow + api + web
docker compose --profile train run --rm trainer   # treina e publica o modelo no registry
```

Web em <http://localhost:5173>, API + Swagger em <http://localhost:8000/docs>, MLflow em <http://localhost:5000>. Até existir um modelo em `production`, a API responde `503` em `/v1/health`. A API usa `data/processed/tracks.csv` para a busca: rode `dvc repro` uma vez antes (veja abaixo).

### Local

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e shared -e "pipeline[dev]" -e "api[dev]"  # shared primeiro: os outros dois dependem dele

mkdir -p data/raw && cp data/processed/tracks_clean.csv data/raw/dataset.csv
dvc add data/raw/dataset.csv
# ...ou, só para exercitar o pipeline (sinal fabricado):
#   python -m resonance_pipeline.synthetic --n 6000 --out data/raw/dataset.csv

cd pipeline && dvc repro && cd ..                      # dados + modelo do zero (~8 min, sqlite local)
cd api && uvicorn app.main:app --reload                # http://localhost:8000/docs
cd web && npm ci && npm run dev                        # http://localhost:5173
```

Testes: `pytest` em `shared/`, `pipeline/` e `api/`; `npm test` em `web/`. Hooks: `pre-commit install`.
O remote do DVC é uma pasta local (`.dvc-storage/`, fora do git); para compartilhar, aponte o remote `minio` (já configurado) para um MinIO/S3.

## Stack

| Camada | Tecnologias |
|---|---|
| Modelo | XGBoost, scikit-learn, SHAP, isotonic calibration, nested CV (StratifiedGroupKFold) |
| Pipeline | DVC, MLflow (registry com alias `production`), pandera, Evidently |
| API | FastAPI, Pydantic, structlog, OpenAPI → tipos TypeScript (openapi-typescript) |
| Web | React 19, TypeScript, Vite, Bootstrap (Sass), Plotly |
| Infra e qualidade | Docker Compose, MinIO, GitHub Actions (CI, treino semanal, drift, docs), pytest, pre-commit |

## Dados e monitoramento

- **Dataset:** Spotify Tracks (Kaggle), 9 features de áudio + `popularity`. Proveniência, licença (não verificada) e vieses em [docs/DATASET_CARD.md](docs/DATASET_CARD.md); dicionário em [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md). O `validate` (pandera) falha cedo com todas as violações juntas.
- **Rigor estatístico:** Mann-Whitney por feature com Bonferroni e FDR; todas diferenciam hits de não-hits, com efeitos pequenos (|rank-biserial| de 0,03 a 0,18) ([`reports/analysis.json`](reports)).
- **Drift (Evidently):** dois relatórios, features vs. treino e distribuição do `percentile_score` vs. referência. `python -m resonance_pipeline.drift --synthetic` roda a demo. Runbook de incidente: [docs/ops/runbook-drift.md](docs/ops/runbook-drift.md); backup: [docs/ops/backup-restore.md](docs/ops/backup-restore.md).

## Limitações e honestidade dos números

- **O histórico do artista é informação de fama, não de áudio.** O ganho vem de saber quem é o artista; o baseline "média do artista" já chega a NDCG 0,733 e o modelo acrescenta ~0,07. Para um artista sem histórico, leia o resultado como "vale ouvir", não "vai estourar" (Spearman 0,14).
- **Circularidade temporal:** o histórico é um retrato do dataset, não da data de lançamento. Não há `release_date`, então não há análise temporal.
- **Viés de seleção:** datasets desse tipo costumam ser montados por sementes de gênero; a proporção de hits por gênero pode ser artefato da coleta.
- **Uma correção que importa:** uma revisão achou vazamento no agrupamento por artista (24,9% das faixas são colaborações). Os números acima são os **depois** da correção; a primeira versão (só áudio, Spearman 0,17) estava inflada.
- **Infra não executada em produção:** Docker, workflows e backup foram escritos e lidos, não rodados de ponta a ponta.

## O que eu faria diferente hoje

- Teria posto o histórico do artista desde o início: o ganho veio de informação nova, não de modelo mais sofisticado.
- Mediria o histórico **no momento do lançamento** e coletaria por amostragem aleatória, para falar de proporção de hits sem ressalva.
- Rodaria o Compose antes de escrever o runbook e o backup.

## Protótipo Streamlit (legado)

A primeira iteração (classificador binário) continua em `app/streamlit_app.py`, com `requirements.txt`, `model/model.pkl` e `data/processed/tracks_clean.csv`, porque o deploy no Streamlit Cloud e o devcontainer apontam para eles. O stack acima não depende dele:

```bash
pip install -r requirements.txt && streamlit run app/streamlit_app.py
```

Os notebooks de `notebooks/` (EDA, gêneros, modelagem, SHAP) documentam essa primeira iteração, com outputs removidos via `nbstripout`.

## Documentação

[ADRs 0001–0008](docs/adr) · [Dataset card](docs/DATASET_CARD.md) · [Performance budget](docs/performance-budget.md) · [Runbook de drift](docs/ops/runbook-drift.md) · [Backup e restore](docs/ops/backup-restore.md) · [CHANGELOG](CHANGELOG.md)
