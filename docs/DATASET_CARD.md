# Dataset card: Spotify Tracks (Kaggle), versão usada no Resonance

## Proveniência
- **Fonte:** dataset público "Spotify Tracks Dataset" do Kaggle: faixas com features de áudio do Spotify e `popularity`.
  O arquivo original **não está no repositório**; o que foi versionado (`data/raw/dataset.csv`, via DVC) é a versão já
  limpa da primeira iteração do projeto (`tracks_clean.csv`, 81.344 linhas), usada aqui como ponto de partida.
- **Licença:** definida pelo autor do dataset no Kaggle; **não verificada** durante este trabalho. Confirme antes de
  redistribuir. As features de áudio e a popularidade são dados do Spotify, sujeitos aos termos da API deles.
- **Data de coleta:** desconhecida; `popularity` é um snapshot estático, ponderado por recência pelo Spotify.
- **Coleta por API (`collect.py`):** não usada. A API do Spotify restringiu os endpoints de playlist e audio features para
  apps em modo de desenvolvimento em fev/2026 ([ADR 0005](adr/0005-mlops-sem-nuvem-paga.md)).

## Conteúdo (após o pipeline)
81.060 faixas, 113 gêneros, 9 features de áudio, `popularity` (0–100), `artist`, `name` e `track_id` opcional. Colunas,
tipos e ranges: [data/DATA_DICTIONARY.md](../data/DATA_DICTIONARY.md), gerado do schema pandera.

Alvo derivado (`target.py`): `popularity_percentile_within_genre` (0–100). Rótulo histórico do badge: `is_hit =
popularity ≥ 60` (10,9% das faixas).

## Colunas derivadas do histórico do artista
`artist_n_tracks`, `artist_mean_popularity` e `artist_hit_share` são calculadas no estágio `target` com as **outras** faixas do mesmo
artista (leave-one-out: nunca a própria linha), e valem NaN quando o artista só tem uma faixa. Como os splits são por artista, essas
colunas não vazam o alvo entre treino e teste. Elas descrevem a **fama** do artista no snapshot do dataset, não o áudio da faixa
([ADR 0008](adr/0008-historico-do-artista-e-duas-modalidades.md)). 17.629 artistas distintos (o **primeiro** artista de cada faixa: colaborações "A;B" contam como A, para que a faixa solo e a colaboração nunca caiam em lados opostos de um split); 9% das faixas têm artista de uma faixa só e ficam sem histórico.

## Limpeza aplicada
Dedup por `track_id` (ou `name + artist + duration_ms` sem id); remoção de nome/artista nulos, faixas > 15 min
(provável podcast/mix) e BPM < 30 (falha de detecção). O `validate` falha cedo com todas as violações de schema juntas.

## Vieses e limites conhecidos
- **Seleção:** datasets desse tipo costumam ser montados varrendo a API por sementes de gênero, não por amostragem
  aleatória das faixas do Spotify. A proporção de hits por gênero pode ser artefato da coleta.
- **Sem tempo:** não há `release_date`; a popularidade não é comparável entre anos e não há análise temporal.
- **Efeito do artista:** a popularidade é em boa parte explicada por quem canta, não pelo áudio; por isso os splits e a CV
  são sempre agrupados por artista.
- **Percentil antes do split:** o alvo (percentil por gênero) é calculado com todas as linhas, inclusive as de teste: efeito de segunda ordem, aceito.
- **Gêneros de nicho:** grindcore, black-metal e study têm ~1.000 faixas e **zero hits** pelo critério `popularity ≥ 60`.
- **Gêneros raros:** 1 gênero tem < 100 faixas (cai em "infrequent" no encoder). Com corte 500, 21 gêneros (7,6% das
  faixas) seriam agrupados (`reports/analysis.json`, `genre_cutoff_sensitivity`).
- **Dado pessoal:** nomes de artistas e faixas públicos; nenhum dado de usuário.

## Dataset sintético para desenvolvimento
`python -m resonance_pipeline.synthetic --n 6000 --out data/raw/dataset.csv` gera um dataset com o mesmo schema para rodar
`dvc repro` sem baixar nada. O sinal é fabricado: serve para testar o pipeline, não para medir o modelo.
