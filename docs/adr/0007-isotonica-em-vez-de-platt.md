# 0007. Calibração isotônica (e não Platt) do `percentile_score`

**Status:** aceito

## Contexto
Com o alvo contínuo não existe `predict_proba` de classificador para calibrar. Existe um **mapeamento** `percentile_score →
P(is_hit histórico)`, e ele não é confiável por padrão: sem calibração, usar `score/100` como probabilidade dá Brier de
0,249, contra 0,097 da taxa-base.

## Decisão
`IsotonicRegression` ajustada entre o `percentile_score` e o rótulo binário histórico, **num fold separado** do treino e do
teste (split de 3 vias por artista: fit 60%, calibração 20%, holdout 20%). Calibrar no dado de treino faria o modelo aprender
o próprio ruído. O Brier antes e depois da calibração é logado no MLflow a cada run; o reliability diagram (pontos) vai em
`reports/evaluation.json`.

## Correção: a entrada da isotônica é a popularidade esperada no gênero, não o percentil
A primeira versão calibrava `percentile_score → is_hit`. Isso falha por construção: o percentil é *intra-gênero* e descarta o gênero,
mas `is_hit` é global (um percentil 90 em pop é um hit; em grindcore, não). Resultado: ROC-AUC do badge de 0,595 e recall de 0,04.

Agora o percentil previsto é convertido na **popularidade esperada no gênero** (o quantil da popularidade do gênero naquele
percentil, tabela de 101 pontos por gênero guardada no bundle) e a isotônica é ajustada sobre isso. Isso devolve ao badge a informação de
gênero sem mudar o ranking. Há **um calibrador por modalidade** ([ADR 0008](0008-historico-do-artista-e-duas-modalidades.md)).
Num teste prévio no mesmo split, só trocar a entrada subiu o ROC-AUC de 0,72 para 0,85.

| Badge (holdout) | ROC-AUC | Brier | ECE | precisão / recall (corte 3:1) |
|---|---|---|---|---|
| Com histórico | 0,884 | 0,062 | 0,014 | 0,51 / 0,66 |
| Só áudio | 0,800 | 0,078 | 0,012 | 0,46 / 0,40 |
| *Antes (percentil direto; medido antes da correção do agrupamento)* | *0,595* | *0,096* | *0,010* | *0,26 / 0,04* |
| Taxa-base constante | n/a | 0,094 / 0,096 | n/a | n/a |

O Brier agora fica claramente abaixo da taxa-base (0,062 e 0,078 contra 0,094 e 0,096): a probabilidade é calibrada **e** informativa.

## Isotônica vs Platt (medido no holdout, versão anterior)
A comparação foi feita com a entrada antiga (percentil), e vale como direção: a diferença é pequena.

| | Brier | ECE |
|---|---|---|
| Isotônica | 0,0957 | 0,0101 |
| Platt (sigmoide) | 0,0956 | 0,0118 |

Escolhemos a isotônica porque (i) com ~16 mil linhas de calibração há dado de sobra para uma curva não-paramétrica; (ii) a relação
score → probabilidade de uma árvore raramente é sigmoide pura; e (iii) ela deu ECE um pouco menor. Com menos dados, Platt seria mais seguro.
