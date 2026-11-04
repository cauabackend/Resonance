# 0005. MLOps sem nuvem paga: DVC + MLflow + MinIO + Compose + Actions

**Status:** aceito

## Contexto
Mostrar versionamento, CI/CD, monitoramento e containerização sem acoplar o projeto a uma conta de nuvem.

## Decisão
| Necessidade | Escolha | Substitui |
|---|---|---|
| Versionar dados/modelo | DVC (remote local; MinIO como segundo remote) | S3 + scripts |
| Registry/tracking | MLflow (sqlite local, ou servidor no Compose) | SageMaker Registry |
| Object storage | MinIO | S3 |
| Orquestração local | Docker Compose | ECS/Fargate |
| Agendamento | GitHub Actions (cron) | EventBridge |
| Imagens | GHCR | ECR |
| Drift | Evidently, relatório HTML como artifact | CloudWatch |

O artefato (`bundle.joblib`: encoder + XGBoost + calibrador + threshold + metadados) é registrado no MLflow Model
Registry com o alias `production`; a API resolve `models:/resonance@production` (ou uma versão fixa via
`MODEL_VERSION_PIN`). Cada versão guarda o MD5 DVC do dataset (lineage) e o commit git.

## Coleta
`collect.py` (API do Spotify) é um pipeline **secundário e não usado** no dataset atual: a coleta real veio do Kaggle,
porque a API restringiu os endpoints de playlist/audio features em fev/2026. Está coberto por testes com mock, mas não
é um estágio do `dvc.yaml`.

## Consequências
`train.yml` (retreino) e `build-and-publish.yml` (imagens) são independentes. Um deploy num VPS é `docker compose pull && up`.

## Por que Docker Compose self-hosted e não um PaaS gerenciado
O Compose sobe `api + web + mlflow + minio` com um comando e roda igual no laptop, no CI (smoke test) e num VPS, sem conta
de nuvem. Um PaaS (Render, Fly.io) entregaria TLS e deploy automático, mas esconderia justamente as peças que o projeto quer
demonstrar (registry, object storage, volumes, healthchecks) e acoplaria o artefato do modelo a um fornecedor. O deploy
num PaaS continua sendo uma extensão natural: as imagens já saem no GHCR.
