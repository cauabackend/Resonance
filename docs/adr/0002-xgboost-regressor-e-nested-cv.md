# 0002. XGBRegressor (e não Random Forest) com nested CV e baselines

**Status:** aceito (substitui a versão que usava um classificador binário; ver [0006](0006-ranking-em-vez-de-classificacao-binaria.md))

## Contexto
O alvo agora é contínuo: `popularity_percentile_within_genre` (0–100). Sem baseline, um NDCG@50 de 0,63 não diz se há
sinal ou só a ordem óbvia do catálogo; e uma CV simples deixa a escolha de hiperparâmetros vazar otimismo para a métrica.

## Decisão
- **Modelo:** `XGBRegressor` no percentil, com `track_genre` como feature (one-hot; gêneros < 100 faixas agrupados como
  "infrequent" pelo próprio `OneHotEncoder`, serializado no mesmo artefato do modelo: gênero inédito não gera skew).
- **Baselines**, registrados no MLflow **antes** do XGBoost: (a) ordenar pela popularidade média do gênero, (b) regressão
  linear (Ridge) nas mesmas features.
- **Nested CV agrupado por artista:** o loop externo (5 folds, `StratifiedGroupKFold`) estima a performance; o interno
  (3 folds) escolhe `max_depth` × `n_estimators`. Splits e CV **sempre por artista**: popularidade é em boa parte efeito do
  artista, e um split por linha vazava (a primeira versão dava ROC-AUC 0,878 contra 0,861 sem vazamento).
- **Gate** (`evaluate.py`): reprova se NDCG@50, Spearman ou Brier piorarem além da tolerância contra o baseline Ridge ou
  contra o modelo em Production, que é reavaliado **no mesmo holdout** do candidato.

## Resultados
Medidos nas duas modalidades ([ADR 0008](0008-historico-do-artista-e-duas-modalidades.md)); holdout separado por artista, NDCG@50 em lotes
aleatórios de 500 faixas.

| | Com histórico: Spearman / NDCG@50 | Só áudio: Spearman / NDCG@50 |
|---|---|---|
| Popularidade média do gênero | −0,008 / 0,497 | −0,008 / 0,497 |
| Regressão linear (só áudio) | n/a | −0,003 / 0,551 |
| Média da popularidade do artista | 0,301 / 0,733 | n/a |
| **XGBRegressor** (holdout) | **0,469 / 0,803** | **0,144 / 0,625** |
| XGBRegressor (nested CV, 5 folds) | 0,491 ± 0,003 / 0,828 ± 0,009 | 0,136 ± 0,009 / 0,623 ± 0,014 |

**Por que XGBoost e não Random Forest:** no leaderboard (abaixo) o RF fica praticamente empatado (0,809 / 0,632), o Hist-GradientBoosting
um pouco atrás e o LambdaMART bem atrás. Ficamos com o XGBoost porque é o melhor ou empatado nas duas modalidades, o `TreeExplainer` é exato
e rápido nele e o tratamento nativo de NaN é o que sustenta as duas modalidades no mesmo modelo.

**O sinal só-áudio é fraco, e isso é o esperado.** Spearman ≈ 0,14 entre o score e o percentil real dentro do gênero: o áudio explica pouco de
quem estoura. O ganho sobre o acaso, porém, é real (NDCG@50 de 0,63 contra 0,50–0,55 dos baselines). O sinal forte vem do histórico do artista.

**Grid.** `max_depth` 6–8 × `n_estimators` 400–800; o loop interno do nested CV escolhe por Spearman médio **das duas modalidades**.

## Leaderboard e estabilidade da explicação
**Leaderboard** (`reports/leaderboard.json`, mesmo split por artista e mesmas features; cada modelo treina em todo o treino, com o histórico
mascarado em 50% das linhas, como em produção):

| | Com histórico: NDCG@50 / Spearman | Só áudio: NDCG@50 / Spearman |
|---|---|---|
| XGBRegressor | 0,811 / 0,476 | 0,635 / 0,143 |
| Random Forest | 0,809 / 0,471 | 0,632 / 0,150 |
| HistGradientBoosting | 0,806 / 0,476 | 0,627 / 0,135 |
| XGBRanker (LambdaMART, `rank:ndcg`) | 0,785 / 0,387 | 0,572 / 0,092 |
| Ridge (com imputação) | 0,786 / 0,457 | 0,547 / 0,010 |

Os três modelos de árvores empatam; o salto sobre o linear é pequeno com histórico (+0,025 de NDCG) e grande só com áudio (0,55 → 0,64), onde as
não-linearidades e a interação com o gênero importam. O LambdaMART fica **pior** que o regressor: com relevância discretizada em 31 níveis e grupos por
gênero ele otimiza uma ordem local que não se traduz no ranking de lotes mistos. LightGBM e CatBoost não entram (dependências novas para modelos que
já empatam com o XGBoost).

**Estabilidade do SHAP entre folds** (5 folds do nested CV, 1.000 faixas por fold): correlação de rank das importâncias médias de 0,96. O fator
`artist_context` domina (9,8 pontos de percentil em média), depois `track_genre` (1,5), `loudness` (1,0), `duration_ms` (0,9) e `energy` (0,8).

## Sensibilidade ao corte de gêneros raros (`features.min_genre_count`)
Com corte 100 só 1 gênero (0,08% das faixas) vira "infrequent"; com 500, 21 gêneros (7,6% das faixas). O percentil é
calculado por gênero original, então o corte afeta só o encoder (que é ajustado no fold de treino, 60% dos dados: lá o corte 100
agrupa 5 gêneros e o modelo em produção conhece 108). Mantivemos 100: um corte maior apagaria gêneros de nicho do modelo.
