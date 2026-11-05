# Resonance

## O problema

Uma gravadora independente com orçamento apertado precisa escolher em quais artistas novos apostar. Contratar A&R experiente custa caro, e intuição sozinha não escala. A pergunta: dá para priorizar quais demos merecem uma escuta atenta?

O modelo não substitui ouvido humano. Ele **ordena**: em vez de ouvir 500 demos, a gravadora separa as ~50 melhores candidatas. Por isso o problema é de **ranking** (percentil de popularidade dentro do gênero), e não de classificação binária global ([ADR 0006](docs/adr/0006-ranking-em-vez-de-classificacao-binaria.md)). O sistema mostra sempre o `percentile_score`, a popularidade esperada no gênero, a probabilidade calibrada do badge, o limiar usado, a versão do modelo e o porquê (SHAP, em pontos de percentil).

## Duas modalidades

O mesmo modelo funciona de dois jeitos ([ADR 0008](docs/adr/0008-historico-do-artista-e-duas-modalidades.md)):

- **Com histórico do artista:** o artista já tem faixas publicadas; a popularidade das outras faixas dele entra no modelo. É o caso de um artista já em catálogo.
- **Só áudio:** uma demo de artista novo, avaliada só pelas características da faixa.

Na tela de detalhe um alternador mostra a mesma faixa nas duas; na API, o histórico é opcional (`artist_mean_popularity` e afins) e `use_artist_context=false` ignora o que foi enviado.

## Arquitetura

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

O backend nunca treina em tempo de requisição: só carrega o artefato versionado (`models:/resonance@production`). Decisões e trade-offs em [docs/adr/](docs/adr).

## Protótipo Streamlit (legado)

O app Streamlit original continua no repositório (`app/streamlit_app.py`, `requirements.txt`, `model/model.pkl` e `data/processed/tracks_clean.csv`)
para o deploy no Streamlit Cloud e o devcontainer, que apontam para esses arquivos. Ele é a primeira iteração (classificador binário) e foi
substituído pelo stack abaixo, que não depende dele:

```bash
pip install -r requirements.txt && streamlit run app/streamlit_app.py
```

## Como rodar

### Tudo com Docker

```bash
cp .env.example .env                      # defina MINIO_ROOT_PASSWORD (obrigatório)
docker compose up --build                 # minio + mlflow + api + web
docker compose --profile train run --rm trainer   # treina e publica o modelo no registry
```

`data/processed/tracks.csv` precisa existir (a API o usa para busca): rode `dvc repro` uma vez, depois de recuperar o dataset (veja "Desenvolvimento local").
Depois: web em <http://localhost:5173>, API + Swagger em <http://localhost:8000/docs>, MLflow em <http://localhost:5000>
e console do MinIO em <http://localhost:9001> (MLflow e MinIO só em `127.0.0.1`). Até existir um modelo em Production a API
responde `503` em `/v1/health`.

### Desenvolvimento local

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e shared -e "pipeline[dev]" -e "api[dev]"  # shared primeiro: é uma dependência dos outros dois

# O remote do DVC é uma pasta local (.dvc-storage/, fora do git); o dataset do protótipo vem no clone:
mkdir -p data/raw && cp data/processed/tracks_clean.csv data/raw/dataset.csv
dvc add data/raw/dataset.csv                           # atualiza o ponteiro .dvc (o hash muda se o git normalizou finais de linha)
# ...ou, só para exercitar o pipeline sem os dados reais (sinal fabricado):
#   python -m resonance_pipeline.synthetic --n 6000 --out data/raw/dataset.csv

cd pipeline && dvc repro && cd ..                      # reproduz dados + modelo do zero (local, sqlite; ~8 min)
cd api && uvicorn app.main:app --reload                # http://localhost:8000/docs
cd web && npm ci && npm run dev                        # http://localhost:5173
```

O dataset bruto (`data/raw/dataset.csv`, Kaggle Spotify Tracks) é versionado pelo DVC (`dataset.csv.dvc`). O remote padrão
(`.dvc-storage/`) é uma pasta local e **não está no GitHub**, então `dvc pull` só funciona na máquina que treinou; para um remote
compartilhado, aponte o remote `minio` (já configurado) para um MinIO/S3 do time. O modelo treinado (`model/`) também não vai para o git:
ele nasce do `dvc repro` e vive no MLflow Registry. Testes: `pytest` em `shared/`, `pipeline/` e `api/`,
`npm test` em `web/`. Pre-commit: `pre-commit install`.

Telas: **Busca** (escolhe uma faixa do catálogo e vê gauge, radar, comparação, valores, um simulador **"E se…"** com
sliders que recalcula o score na API, e "por que essa decisão"); **Ranking de demos** (`/ranking`: sobe um CSV, recebe o lote
ordenado com `rank`, modalidade e barra de percentil por linha, o corte do top-50, e exporta em CSV;
[exemplo](web/public/exemplo-demos.csv), metade com histórico do artista); **Model card** (`/model-card`: métricas por modalidade e
limites de uso); e um toggle de **alto contraste**. Erros de API oferecem "Tentar de novo".

Para rodar sem baixar dados: `python -m resonance_pipeline.synthetic --n 6000 --out data/raw/dataset.csv` gera um dataset
sintético com o mesmo schema (o sinal é fabricado: só serve para exercitar o pipeline).

## Resultados

Todos os números vêm de **splits e CV agrupados por artista** (o mesmo artista nunca está em treino e teste; colaborações "A;B" contam
como o primeiro artista) e de `reports/evaluation.json`. Dataset: 81.060 faixas, 113 gêneros, 17.629 artistas. NDCG@50 é calculado em lotes
aleatórios de 500 faixas (100 lotes do holdout): "o top-50 do modelo coincide com o top-50 real?". A modalidade **com histórico** é medida nas
faixas cujo artista tem outras faixas no dataset (91% do holdout); as demais caem em **só áudio**, como na API.

| Holdout por artista | Spearman | NDCG@50 | ROC-AUC do badge | Piso (baseline): Spearman / NDCG@50 |
|---|---|---|---|---|
| **Com histórico do artista** | **0,469** | **0,803** | **0,884** | média do artista: 0,301 / 0,733 |
| Só áudio (artista novo) | 0,144 | 0,625 | 0,800 | regressão linear: −0,003 / 0,551 |
| Nested CV, com histórico | 0,491 ± 0,003 | 0,828 ± 0,009 | | |
| Nested CV, só áudio | 0,136 ± 0,009 | 0,623 ± 0,014 | | |

- **Com o histórico do artista o ranking triplica** (Spearman 0,14 → 0,47). Num lote aleatório de 500 faixas, as 50 que o modelo escolhe têm
  percentil real médio de **76,1** (acaso: 50); só com áudio, 58,3. No 1% de maior score, 41% são hits com histórico (38% só com áudio; a taxa-base é 10,6%).
- **Os scores deixam de ficar perto de 50 quando há histórico:** desvio 14,1 (p1 = 22, p99 = 84), contra 5,6 só com áudio (p5 = 41, p95 = 58,5).
- **Só áudio continua fraco, e é o teto do tipo de feature**, não um bug (seção abaixo). As features novas de áudio e a normalização por gênero
  não mudam isso de forma mensurável; o que melhorou nessa modalidade foi o badge, ao calibrar pela popularidade esperada **no gênero**
  ([ADR 0007](docs/adr/0007-isotonica-em-vez-de-platt.md)).
- **Badge "é hit"** (corte de custo 3:1): com histórico, precisão 0,51 e recall 0,66; só áudio, 0,46 e 0,40 (taxa-base de hits: 10,6%).
  Calibração: ECE 0,014 / 0,012, Brier 0,062 / 0,078 contra 0,094 / 0,096 da taxa-base.
- **Gêneros de nicho** (zero hits cada, mas com percentil), Spearman com histórico: grindcore 0,28, black-metal 0,45, study 0,34 (só áudio: 0,06 / 0,11 / 0,23).
- **O artista domina a explicação:** `artist_context` pesa 9,8 pontos de percentil no SHAP em média, contra 1,5 do gênero e ≤ 1,0 de qualquer feature de áudio
  (importância estável entre os folds: correlação de ranks 0,96).
- **Leaderboard** (`reports/leaderboard.json`, NDCG@50 com histórico / só áudio): XGBRegressor 0,811 / 0,635, HistGradientBoosting 0,806 / 0,627,
  Random Forest 0,809 / 0,632, XGBRanker (LambdaMART) 0,785 / 0,572, Ridge 0,786 / 0,547 ([ADR 0002](docs/adr/0002-xgboost-regressor-e-nested-cv.md)).
  Com histórico os modelos de árvores quase empatam entre si e com o Ridge (0,786).

**O que esses números significam e o que não.** O histórico do artista é informação de **fama**, não de áudio: o ganho enorme vem de saber quem é o
artista, e o baseline "média do artista" já chega a NDCG 0,733 (o modelo acrescenta ~0,07). Para um artista realmente sem histórico, só áudio,
o modelo é útil para priorizar, não para prever. A avaliação com histórico mede "dado o histórico de um artista, quão bem ranqueamos uma faixa
dele", e não "prever o sucesso de um desconhecido" ([ADR 0008](docs/adr/0008-historico-do-artista-e-duas-modalidades.md)).

**Uma correção que importa.** Uma revisão antes do primeiro commit achou que o agrupamento por artista usava o texto completo de `artists`
(`"A;B"` e `"A"` eram grupos diferentes): 24,9% das faixas são colaborações, então a faixa solo de A caía no treino e a colaboração no teste. Os
números acima são os **depois** da correção; os da primeira versão (só áudio: Spearman 0,17) estavam inflados por esse vazamento.

### Rigor estatístico (`reports/analysis.json`)

Mann-Whitney U por feature contra o rótulo histórico, com Bonferroni e FDR sobre as 9 features: **todas** diferenciam
hits de não-hits (p corrigido < 1e-5), mas com tamanhos de efeito pequenos (|rank-biserial| de 0,03 a 0,18; maiores:
`instrumentalness`, `loudness`, `danceability`). "BPM não importa" do EDA original não se sustenta como "sem efeito": é
pequeno (−0,05). VIF: `energy` 4,1 e `loudness` 3,3 (correlacionadas, abaixo de 5).

## Desempenho da API

`python api/scripts/benchmark.py` (máquina de desenvolvimento, CPU, 1 worker uvicorn):

| Endpoint | Resultado |
|---|---|
| `POST /v1/predict` (n = 200, sequencial, com SHAP) | p50 20,6 ms · p95 22,2 ms · ~47 faixas/s |
| `POST /v1/predict`, 8 clientes simultâneos (n = 2.000) | p50 132 ms · p95 173 ms · p99 198 ms · 57 req/s |
| `POST /v1/predict/batch` (500 faixas, ordenado) | 36 ms · ~13.800 faixas/s |

Suba a API com `RATE_LIMIT_PER_MINUTE=0` para o benchmark (o limite padrão é 120 req/min por IP). O orçamento de performance
(p95 < 200 ms, verificado no CI) está em [docs/performance-budget.md](docs/performance-budget.md).

## O teto do áudio puro e o viés do dataset

**Teto.** A literatura de *Hit Song Science* (que tenta prever sucesso musical só por features sonoras) encontra um teto
baixo: boa parte do que faz uma música estourar não está no áudio, está em quem canta, na distribuição e no marketing
(Pachet & Roy, "Hit Song Science is Not Yet a Science", ISMIR 2008; Herremans, Martens & Sörensen, "Dance Hit Song
Prediction", *Journal of New Music Research*, 2014). O plano de revisão cita ~70–75% de acurácia balanceada como teto típico
(não verifiquei esse número nos artigos). O resultado só áudio deste projeto (ρ ≈ 0,14) é coerente com isso, e **o que moveu o
ponteiro foi justamente informação fora do áudio**: o histórico do artista (enriquecimento nº 1 do plano). Ainda faltam, em ordem de custo/benefício:
(2) ano de lançamento e "era", (3) seguidores/popularidade do artista **no momento do lançamento** (o histórico daqui é um retrato do
dataset), e (4) mais features compostas. Os itens 2 e 3 exigem dados que o dataset do Kaggle não tem e que a API do Spotify deixou de fornecer a
apps em modo de desenvolvimento; **não foram implementados**.

**Viés de seleção.** Datasets desse tipo costumam ser montados varrendo a API do Spotify por *seeds de gênero* (uma playlist
"hip-hop", uma "grindcore"), não por amostragem aleatória das faixas do Spotify. A proporção de hits por gênero no dataset
pode, portanto, não refletir o mercado real: "hip-hop tem mais hits que grindcore" pode ser artefato da coleta. O percentil
por gênero atenua o efeito entre gêneros, mas não dentro de cada um.

## Dados e qualidade

Dicionário de dados gerado do schema pandera: [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md). O `validate` falha cedo e
com todas as violações juntas se o CSV mudar de formato (já pegou uma linha com nome nulo). Dedup por `track_id` (ou
`name + artist + duration_ms`), faixas > 15 min e BPM < 30 removidos. O histórico do artista é leave-one-out (só as outras faixas),
e [docs/DATASET_CARD.md](docs/DATASET_CARD.md) documenta proveniência e vieses.

## Monitoramento

O job de drift (Evidently) gera **dois** relatórios: features recebidas vs. treino (`reports/drift.html`) e distribuição do
`percentile_score` previsto vs. a referência (`reports/drift_predictions.html`), que pega o drift de predição mesmo sem
mudança nas features (só na modalidade só-áudio, a das demos: misturar as duas geraria falso alarme).
`python -m resonance_pipeline.drift --synthetic` roda a demo contra tráfego sintético deslocado (os dois relatórios disparam alerta).
Faixas abertas pelo catálogo na UI (`source: "catalog"`) não entram no log.

## Operação e documentação
- [docs/DATASET_CARD.md](docs/DATASET_CARD.md): proveniência, licença (não verificada), vieses.
- [docs/ops/runbook-drift.md](docs/ops/runbook-drift.md): triagem de um incidente fictício de "scores mais baixos".
- [docs/ops/backup-restore.md](docs/ops/backup-restore.md): backup e restore do MinIO e do MLflow (não executado aqui, sem Docker).
- `GET /v1/health/deep` compara o hash do modelo carregado com o artefato que o registry serve; `MODEL_SHA256` recusa um
  bundle adulterado antes de desserializá-lo. [CHANGELOG.md](CHANGELOG.md) registra o contrato da API; o OpenAPI é publicado
  em GitHub Pages (`docs.yml`).

## O que eu faria diferente hoje
- **Teria posto o histórico do artista desde o início.** A primeira versão tentou espremer sinal de 9 features de áudio e entregou scores
  sempre perto de 50; o ganho veio de informação nova, não de modelo mais sofisticado.
- **Eu mediria o histórico do artista no momento do lançamento,** e não o retrato do dataset: com a popularidade atual, parte da fama do artista é
  consequência do sucesso da própria faixa. O leave-one-out evita o vazamento direto, mas não essa circularidade temporal.
- **Teria começado com um dataset coletado por amostragem aleatória**, e não por sementes de gênero, para poder falar de
  proporção de hits sem ressalva.
- **Teria medido memória e rodado o Compose antes de escrever o runbook e o backup**: a infra (Docker, workflows, backup)
  aqui foi escrita e lida, não executada.

## Limitações

O modelo usa features de áudio e o histórico do artista. Não sabe se uma música viralizou, se entrou em novela, nem se houve
marketing. A popularidade do Spotify é um snapshot ponderado por recência e o dataset não tem data de lançamento, o que
impede análise temporal. O app é um filtro inicial, não uma decisão final. **Para um artista sem nenhum histórico o modelo só enxerga o
áudio, e aí o sinal é fraco** (Spearman 0,14): leia o resultado como "vale ouvir", não como "vai estourar".

A coleta via API do Spotify (`pipeline/src/resonance_pipeline/collect.py`) não é usada: a API restringiu os endpoints de
playlist/audio features em fev/2026 e o dataset veio do Kaggle ([ADR 0005](docs/adr/0005-mlops-sem-nuvem-paga.md)).

## Notebooks

`notebooks/` guarda a análise original (EDA, gêneros, modelagem, SHAP, erros), com outputs removidos via `nbstripout`
(o `02_eda.ipynb` tinha 10 MB). São a documentação histórica da primeira iteração (classificação binária); a lógica
repetida entre eles **não** foi extraída para `pipeline/src/`. Os números atuais vivem em `reports/`, gerados pelo pipeline.
