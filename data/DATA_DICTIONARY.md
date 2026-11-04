# Dicionário de dados

Gerado de `pipeline/src/resonance_pipeline/schema.py` (`dvc repro validate`). Não editar à mão.

| Coluna | Tipo | Range válido | Nota de qualidade |
|---|---|---|---|
| `name` | str | — | dedup por track_id; sem id, name+artist+duration_ms |
| `artist` | str | — | primeiro artista listado; também é o grupo dos splits (sem vazamento entre artistas) |
| `popularity` | int64 | 0–100 | recency-weighted, snapshot estático; não comparável entre anos (sem release_date) |
| `track_genre` | str | — | categorias raras (< 100 exemplos) agrupadas pelo encoder |
| `liveness` | float64 | 0–1 | probabilidade de ser gravação ao vivo (Spotify) |
| `key` | int64 | -1–11 | tom (0-11, -1 = não detectado) |
| `mode` | int64 | {0, 1} | 0 = menor, 1 = maior |
| `time_signature` | int64 | 0–7 | compasso estimado (0-7) |
| `explicit` | bool | — | conteúdo explícito (sinal de letra, disponível para o demo) |
| `track_id` | str | — | id do Spotify (opcional) |
| `danceability` | float64 | 0–1 | sem outliers conhecidos |
| `energy` | float64 | 0–1 | sem outliers conhecidos |
| `valence` | float64 | 0–1 | sem outliers conhecidos |
| `speechiness` | float64 | 0–1 | sem outliers conhecidos |
| `acousticness` | float64 | 0–1 | sem outliers conhecidos |
| `instrumentalness` | float64 | 0–1 | sem outliers conhecidos |
| `loudness` | float64 | -60–5 | correlacionada com energy (VIF em reports/analysis.json) |
| `tempo` | float64 | ≥ 30 | valores próximos de 0 indicam falha de detecção de BPM; tratados como outlier (< 30) |
| `duration_ms` | int64 | 1–900000 | faixas > 15 min são provavelmente podcast/mix; filtradas |
| `popularity_percentile_within_genre` | float64 | 0–100 | alvo real de treino: percentil (0-100) da popularidade dentro do próprio gênero; recalculado em target.py |
| `is_hit` | int64 | {0, 1} | rótulo histórico (popularity ≥ 60); NÃO é alvo de treino, só calibra a probabilidade do badge |
| `artist_n_tracks` | float64 | ≥ 0 | OUTRAS faixas do mesmo artista no dataset (leave-one-out: nunca inclui a própria linha) |
| `artist_mean_popularity` | float64 | 0–100 | popularidade média das OUTRAS faixas do artista; NaN se o artista só tem esta faixa |
| `artist_hit_share` | float64 | 0–1 | fração das OUTRAS faixas do artista com popularity ≥ 60 |
