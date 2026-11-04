# 0006. Ranking por gênero, e não classificação binária global

**Status:** aceito. **Reverte** a decisão anterior do projeto (classificador binário com `is_hit = popularity ≥ 60` e `scale_pos_weight`).

## Contexto
O README promete "separar as 50 demos que valem escuta de 500". Isso é um problema de **ranking**. O pipeline original
(`is_hit = popularity ≥ 60`, classificador) resolvia outro: um corte global ignora que "60 de popularidade" é trivial num
gênero e quase inatingível em outro, e um classificador binário só diz sim/não, sem produzir uma ordem.

## Decisão
- **Alvo:** `popularity_percentile_within_genre` (`target.py`): um percentil 90 em grindcore passa a significar o mesmo que
  um 90 em pop.
- **Modelo:** `XGBRegressor` no percentil contínuo. O score previsto é o `percentile_score`.
- **Métricas primárias:** NDCG@50 em lotes aleatórios de 500 faixas (o caso de uso: o top-50 do modelo contra o top-50 real
  do lote) e correlação de Spearman. ROC-AUC/F1 deixam de ser primárias.
- **Badge:** camada derivada (ADR 0003/0007), nunca o alvo de treino.
- **API:** `/v1/predict/batch` devolve o lote **ordenado** por `percentile_score`, com `rank` por faixa.

## O custo, medido (e como foi mitigado)
Um percentil intra-gênero remove, por construção, a informação de **qual gênero** é a faixa, e o gênero é a feature que mais separa hits
globais. Na primeira versão do ranking isso derrubou o ROC-AUC do badge de 0,861 (classificador) para 0,595. Dois ajustes o recuperaram sem
abrir mão do ranking: a calibração passou a usar a **popularidade esperada no gênero** ([ADR 0007](0007-isotonica-em-vez-de-platt.md);
ROC-AUC 0,800 só com áudio e 0,884 com o histórico do artista) e o modelo ganhou o contexto do artista
([ADR 0008](0008-historico-do-artista-e-duas-modalidades.md)).

## Consequências
Um lote misto de gêneros é ordenado pelo percentil previsto, o que trata gêneros diferentes como comparáveis na escala
0–100. É a premissa do produto, não uma verdade medida.

## Por que "nenhuma música recebe nota alta"
O sintoma original (scores sempre medianos, nunca um "90%") tinha várias causas: um peso de classe mal configurado
([ADR 0004](0004-scale-pos-weight-e-o-bug-do-0.65.md)), dedup que deixava duplicatas, ausência de normalização por gênero, falta de
features de contexto e um dataset com variância achatada. O enquadramento por ranking e a normalização por gênero resolvem as três
primeiras; as duas últimas são limites de informação, e a mais importante só se resolve **trazendo informação nova**:

- **Só áudio:** desvio do score de 5,6 (p5 = 41, p95 = 58,5); com Spearman ≈ 0,14 o regressor "encolhe" as previsões para a média. O score
  ordena, mas não é uma nota absoluta: isso é o teto do áudio puro.
- **Com o histórico do artista:** desvio de 14,1 (p1 = 22, p99 = 84) e 41% de hits no 1% de maior score (10,6% no geral). Os scores passam a usar a escala
  ([ADR 0008](0008-historico-do-artista-e-duas-modalidades.md)).
