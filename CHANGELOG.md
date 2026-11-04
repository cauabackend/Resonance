# Changelog

Versionamento semântico da API (`/v1`). Uma quebra de contrato cria `/v2` e mantém o `/v1` por um ciclo de depreciação; o
contrato é o OpenAPI (`web/openapi.json`, publicado em GitHub Pages) e os tipos do front são gerados dele.

## [1.2.0] — 2026-10-29
### Adicionado
- Duas modalidades no mesmo modelo: `artist_aware` (a faixa traz o histórico do artista) e `audio_only`. `PredictionResponse` ganha
  `mode` e `expected_popularity` (popularidade esperada no gênero para o percentil previsto); `BatchItem` ganha `mode` e `expected_popularity`.
- `TrackFeatures` aceita, opcionalmente: `liveness`, `key`, `mode`, `time_signature`, `explicit` e `artist_n_tracks`,
  `artist_mean_popularity`, `artist_hit_share`.
- `PredictionRequest.use_artist_context` e `BatchRequest.use_artist_context` (padrão `true`): `false` ignora o histórico enviado.
### Alterado
- `BatchPredictionResponse.threshold` virou `thresholds` (um por modalidade). **Quebra para clientes do lote.** O badge usa o calibrador e o
  threshold da modalidade de cada faixa.
- `ModelMetadata` ganha `baseline_artist_mean_*`; `metrics` traz também as métricas `audio_only_*`.
- O log de predição guarda `mode`; o drift de predição compara só a modalidade `audio_only`.

## [1.1.0] — 2026-10-29
### Adicionado
- `GET /v1/health/deep`: compara o hash (sha256) do modelo carregado com o artefato que o registry serve agora.
- `MODEL_SHA256`: recusa um bundle com hash diferente do esperado antes de desserializá-lo.
- Rate limit por IP em memória (`RATE_LIMIT_PER_MINUTE`, 120 por padrão; `0` desliga). Resposta `429` com `Retry-After`.
- `PredictionRequest.source` (`user` | `catalog`): faixas do catálogo não entram no log de drift.
- Log de predição inclui `percentile_score`; novo relatório de drift de predição.
### Alterado
- `TrackFeatures` agora rejeita NaN/Infinity e valores fora dos ranges de treino (`tempo` 30–300, `loudness` −60–5,
  `duration_ms` ≤ 900000, `track_genre` ≤ 64 caracteres) e campos extras. **Pode quebrar clientes que enviavam valores fora do range.**

## [1.0.0] — 2026-10-29
### Alterado (quebra em relação ao protótipo Streamlit)
- O problema passou de classificação binária para **ranking por gênero**: `PredictionResponse` traz `percentile_score`,
  `is_hit` (derivado), `probability` (isotônica), `threshold`, `model_version`, `top_factors` e `factors` (SHAP em pontos de percentil).
- `POST /v1/predict/batch` devolve `BatchPredictionResponse`: `results` ordenado por `percentile_score`, com `rank` e
  `input_index`, e `rank[i]` alinhado à lista enviada.
### Adicionado
- `/v1/predict`, `/v1/predict/batch`, `/v1/tracks/search`, `/v1/tracks/{id}`, `/v1/model/metadata`, `/v1/health`.
