# Backup e restore: MinIO e MLflow

O que precisa sobreviver a um desastre: **artefatos dos modelos e dos dados** (MinIO: buckets `models` e `dvc`) e os
**metadados de experimento e do registry** (MLflow: `sqlite:////mlruns/mlflow.db`, no volume `mlflow-data`). Sem o banco, os
artefatos viram arquivos soltos; sem os artefatos, o banco aponta para o vazio.

Os comandos abaixo assumem o Compose deste repositório, rodando a partir da raiz. Não foram executados neste ambiente (sem
Docker): teste o restore antes de depender dele.

## Backup

```bash
mkdir -p backup && STAMP=$(date +%Y%m%d-%H%M%S)

# 1) MLflow (SQLite): cópia consistente com o servidor parado
docker compose stop mlflow
docker run --rm -v resonance_mlflow-data:/mlruns -v "$PWD/backup":/backup alpine \
  tar czf /backup/mlflow-$STAMP.tgz -C /mlruns .
docker compose start mlflow

# 2) MinIO (buckets models e dvc) com o cliente mc
docker run --rm --network resonance_default -v "$PWD/backup/minio-$STAMP":/out \
  -e MC_HOST_src="http://$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD@minio:9000" minio/mc \
  mirror src/models /out/models && \
docker run --rm --network resonance_default -v "$PWD/backup/minio-$STAMP":/out \
  -e MC_HOST_src="http://$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD@minio:9000" minio/mc \
  mirror src/dvc /out/dvc
```

O nome da rede e dos volumes depende do nome do projeto do Compose (`resonance_` é o padrão para o diretório `Resonance`).
Os dados versionados também vivem no remote DVC: `dvc push` é um segundo backup do dataset e dos modelos.

## Restore

```bash
docker compose down                       # sem apagar volumes (sem -v)
docker volume rm resonance_mlflow-data && docker volume create resonance_mlflow-data
docker run --rm -v resonance_mlflow-data:/mlruns -v "$PWD/backup":/backup alpine \
  tar xzf /backup/mlflow-<STAMP>.tgz -C /mlruns
docker compose up -d minio minio-init
# devolve os objetos aos buckets
docker run --rm --network resonance_default -v "$PWD/backup/minio-<STAMP>":/in \
  -e MC_HOST_dst="http://$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD@minio:9000" minio/mc mirror /in/models dst/models
docker run --rm --network resonance_default -v "$PWD/backup/minio-<STAMP>":/in \
  -e MC_HOST_dst="http://$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD@minio:9000" minio/mc mirror /in/dvc dst/dvc
docker compose up -d
```

## Verificação pós-restore
1. `curl localhost:5000/health` e o alias `production` existe: `models:/resonance@production`.
2. `curl localhost:8000/v1/health/deep` devolve `status: ok` (o hash carregado bate com o que o registry serve).
3. `dvc pull` recupera o dataset do remote **se ele for um remote compartilhado** (o padrão, `.dvc-storage/`, é uma pasta local da máquina).

## Política sugerida
Backup diário do MLflow e do MinIO com retenção de 7 dias; teste de restore trimestral num ambiente descartável. Para
volume maior, trocar o SQLite por PostgreSQL e usar a replicação do próprio MinIO.
