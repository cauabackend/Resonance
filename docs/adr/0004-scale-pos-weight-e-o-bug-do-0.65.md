# 0004. `scale_pos_weight`: o bug do `ratio * 0.65` e por que ele deixou de existir

**Status:** aceito

## Contexto
O `train.py` original usava `scale_pos_weight = ratio * 0.65`. O peso natural da classe positiva seria `ratio` (não-hits por
hit); multiplicar por 0,65 sub-corrige o desbalanceamento de propósito, empurrando a fronteira para a classe majoritária.
É por isso que hits óbvios saíam com score baixo. Não era só um número sem justificativa: era um bug de configuração.

## Decisão
1. **Corrigir antes de calibrar**, nunca depois: calibrar por cima de um peso arbitrário mascararia o bug.
2. Na primeira versão corrigida, o peso passou por um grid documentado (`ratio × {0,3; 0,65; 1,0}`, CV agrupada, critério
   *average precision*). O melhor foi o **menor** multiplicador (0,3; AP 0,465 contra 0,446 em 1,0), com tendência
   monotônica, o oposto do que o `0,65` fixo sugeria.
3. Com a mudança de enquadramento para ranking ([0006](0006-ranking-em-vez-de-classificacao-binaria.md)) o alvo é contínuo e
   **não há classe positiva para pesar**: `scale_pos_weight` deixa de existir. O desbalanceamento (10,9% de hits) só reaparece
   na camada do badge, onde a isotônica é ajustada sobre o rótulo histórico e o custo 3:1 trata a assimetria explicitamente.

## Consequências
Os hiperparâmetros que sobram (`max_depth`, `n_estimators`) são escolhidos pelo loop interno do nested CV (ADR 0002).
