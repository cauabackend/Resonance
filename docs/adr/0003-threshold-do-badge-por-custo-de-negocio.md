# 0003. Threshold do badge por custo de negócio

**Status:** aceito

## Contexto
O badge "Potencial de hit / Baixo potencial" é uma camada derivada do `percentile_score`, não o alvo de treino. Um
`threshold = 0.5` é um padrão sem justificativa. Para uma gravadora, deixar passar um hit (falso negativo) custa mais que
investigar uma demo que não vinga (falso positivo).

## Decisão
O threshold (sobre a probabilidade calibrada) minimiza `fn_cost · FN + fp_cost · FP` no fold de calibração, com
`fn_cost = 3` e `fp_cost = 1` (`pipeline/params.yaml`). Como a probabilidade é calibrada, o ótimo teórico é
`p ≥ fp/(fp+fn) = 0,25`. A API devolve o `threshold` em cada resposta e o gauge o desenha.

## Resultado
Um threshold por modalidade, cada um sobre o seu calibrador ([ADR 0007](0007-isotonica-em-vez-de-platt.md)). No holdout:
**com histórico**, precisão 0,51 e recall 0,66; **só áudio**, precisão 0,46 e recall 0,40. A taxa-base de hits é 10,6%, então o badge
acerta cerca de 4 a 5 vezes mais que o acaso. (Antes da calibração por gênero, o badge tinha precisão 0,26 e recall 0,04: marcava quase nada.)

## Sensibilidade ao corte de popularidade
`is_hit = popularity ≥ 60` é uma convenção. `reports/analysis.json` traz a prevalência de hits para 55/60/65/70
(17,3% / 10,9% / 6,3% / 3,3%) e o AUC de uma logística em cada corte.

## Consequências
Os custos 3:1 são uma premissa de negócio, não um fato medido: mudar `params.yaml` e rodar `dvc repro` recalcula tudo.
