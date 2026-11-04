# Runbook (exercício): "o modelo em produção passou a dar scores mais baixos"

> **Incidente fictício**, montado com o relatório de drift de demonstração (`python -m resonance_pipeline.drift
> --synthetic`), que desloca `danceability` e `energy` do tráfego sintético. Os números abaixo são dessa demo.

## Sintoma
O time de A&R reclama: "as demos desta semana estão todas com percentil baixo". O percentil médio previsto passou de ~50
para ~43.

## Triagem

1. **O serviço está de pé e é o modelo certo?**
   `GET /v1/health/deep` → `status: ok` e `loaded_sha256 == registry_sha256` descartam artefato trocado ou adulterado. Se
   vier `mismatch`, o registry serve outro modelo que a API não carregou: reinicie a API (ou fixe `MODEL_VERSION_PIN`).
2. **Mudou o modelo?** Compare `model_version` das respostas e o hash do dataset em `/v1/model/metadata` com a última
   promoção no MLflow (`models:/resonance@production`). Se houve retreino, o gate (`reports/evaluation.json`) mostra por
   que foi aprovado.
3. **Mudaram as entradas (drift de dado)?** Abra `reports/drift.html`. Na demo: 3 de 9 features driftadas (33%, limiar de
   alerta 20%): `danceability` e `energy` com KS p < 1e-30. O catálogo avaliado mudou: é entrada diferente, não modelo
   quebrado.
4. **Mudou só a saída (drift de predição)?** Abra `reports/drift_predictions.html`. Na demo, a média do `percentile_score`
   foi de 50,2 (referência) para 42,7. Se as features NÃO driftaram e o score sim, suspeite do modelo ou do pré-processamento
   (encoder de gênero, versão de biblioteca).
5. **Os logs são confiáveis?** O log de predição (`data/prediction_logs/dt=…`) só tem requisições `source: "user"`; a
   navegação no catálogo da UI não entra. Se o volume cair a zero, o front pode estar enviando `source: "catalog"`.

## Decisão

| Evidência | Conclusão | Ação |
|---|---|---|
| Só drift de dado | O lote avaliado é diferente do treino | Avisar o time; se o novo perfil é permanente, retreinar (`train.yml`) com dados novos |
| Drift de dado + de predição | O modelo reage ao novo perfil como esperado | Idem; conferir NDCG@50 num lote rotulado, se existir |
| Só drift de predição | Suspeita de modelo/pipeline | Reverter para a versão anterior (`MODEL_VERSION_PIN=<n>`), abrir incidente |
| `mismatch` no health profundo | API desatualizada ou artefato trocado | Reiniciar a API; investigar quem escreveu no registry |

## Depois
Registrar o incidente, anexar os HTMLs do Evidently (são artifacts do workflow `drift.yml`) e, se o retreino for a saída,
lembrar que o gate só promove o candidato que não piorar NDCG@50, Spearman e Brier.
