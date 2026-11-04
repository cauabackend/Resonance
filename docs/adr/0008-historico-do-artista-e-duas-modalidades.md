# 0008. Histórico do artista como contexto: duas modalidades no mesmo modelo

**Status:** aceito

## Contexto
Com só 9 features de áudio o ranking dentro do gênero tem Spearman ≈ 0,14: o áudio explica pouco de quem estoura, e os scores
ficam espremidos perto de 50 ([ADR 0006](0006-ranking-em-vez-de-classificacao-binaria.md)). O maior sinal que o dataset já
tinha e que o modelo ignorava é **quem é o artista**: a popularidade das outras faixas dele. É o enriquecimento nº 1 do plano de
revisão (fama do artista) e o nº 2 (histórico de hits).

## Decisão
Três números do histórico do artista entram no modelo, sempre **leave-one-out** (calculados só com as OUTRAS faixas do artista,
nunca com a própria linha):

- `artist_n_tracks`, `artist_mean_popularity`, `artist_hit_share`;
- mais `artist_pct_in_genre`, derivado deles e do gênero da faixa: o percentil que aquela popularidade média teria no gênero.

Além disso: 5 features de áudio que o dataset já trazia e o modelo não usava (`liveness`, `key`, `mode`, `time_signature`,
`explicit`), cada feature de áudio normalizada **dentro do gênero** (z-score) e `energy × danceability`.

**Duas modalidades no mesmo modelo.** O XGBoost trata NaN nativamente. No treino o histórico é apagado em 50% das linhas
(`features.artist_mask_rate`), então o modelo aprende a prever **com** o histórico (artista que já tem faixas) e **sem** ele (demo
de artista novo). A API escolhe pela entrada: `artist_mean_popularity` presente → `artist_aware`; ausente, ou
`use_artist_context=false` → `audio_only`. Cada modalidade tem o seu calibrador e o seu threshold de badge.

**Mesmo código no treino e na API.** A engenharia de features vive em `shared/resonance_features`, instalada pelos dois lados:
sem train/serve skew e sem classe customizada dentro do pickle (a API não importa o pipeline).

## Por que isso não vaza o alvo
- O histórico é leave-one-out: a popularidade da própria faixa nunca entra na própria feature.
- O split e a CV são **por artista**: nenhum artista aparece em treino e teste. No teste, o histórico de um artista vem das outras
  faixas dele que também estão no teste: é exatamente o cenário de uso (faixas já publicadas dele com popularidade conhecida).
- O que isso mede é "dado o histórico de um artista, quão bem ranqueamos uma faixa dele"; **não** mede "prever o sucesso de um
  artista desconhecido". Para esse caso existe a modalidade só-áudio, avaliada à parte.

## Correção encontrada na revisão: "artista" é o primeiro artista
`artists` traz colaborações como `"A;B"`. A primeira versão agrupava por esse texto inteiro: 24,9% das faixas são colaborações (31.362 "grupos" para
17.629 artistas principais), então a faixa solo de A podia ir para o treino e a colaboração A;B para o teste (vazamento) e o histórico de A ficava
incompleto. Agora `preprocess` usa o **primeiro artista** como grupo, tanto dos splits e da CV quanto do histórico. Os números abaixo são os
depois da correção; a primeira versão tinha Spearman só-áudio 0,174, inflado por esse vazamento.

## Resultado (holdout por artista; nested CV entre parênteses)
A modalidade com histórico é medida onde ela se aplica: faixas cujo artista tem outras faixas no dataset (91% do holdout; as demais vão para
só áudio, como na API), e é nessa população que o seu calibrador e o threshold do badge são ajustados.

| | Spearman | NDCG@50 | ROC-AUC do badge | Piso (baseline) |
|---|---|---|---|---|
| **Com histórico do artista** | **0,469** (0,491 ± 0,003) | **0,803** (0,828 ± 0,009) | **0,884** | média do artista: 0,301 / 0,733 |
| Só áudio (artista novo) | 0,144 (0,136 ± 0,009) | 0,625 (0,623 ± 0,014) | 0,800 | regressão linear: −0,003 / 0,551 |

- Com histórico, o ranking **triplica** (Spearman 0,14 → 0,47). Num lote aleatório de 500 faixas, as 50 que o modelo escolhe têm percentil real médio de
  **76,1** (acaso = 50); só áudio, 58,3.
- Os scores **deixam de ficar em 50**: com histórico o desvio é 14,1 (p1 = 22, p99 = 84), contra 5,6 só com áudio. No 1% de maior score, 41% são hits com
  histórico (10,6% no geral).
- Gêneros de nicho (Spearman com histórico): grindcore 0,28, black-metal 0,45, study 0,34 (só áudio: 0,06 / 0,11 / 0,23).
- O artista domina a explicação: o fator `artist_context` pesa em média **9,8 pontos de percentil** no SHAP, contra 1,5 do gênero e ≤ 1,0 de qualquer
  feature de áudio.

## O que isso NÃO é
- **Não melhorou a modalidade só-áudio.** Corrigido o vazamento, o áudio puro dá Spearman 0,144: o teto continua valendo, e as features novas de áudio
  e a normalização por gênero não o mudam de forma mensurável. O ganho do badge nessa modalidade (ROC-AUC 0,800) vem da calibração por gênero
  ([ADR 0007](0007-isotonica-em-vez-de-platt.md)), não do ranking.
- **O baseline "média do artista" já é forte** (NDCG 0,733; uma regressão linear com as mesmas features chega a 0,786). O modelo acrescenta ~0,07 de NDCG
  sobre a média do artista e ~0,02 sobre a linear: o grosso do ganho é a informação do artista, não a complexidade do modelo.
- **O histórico precisa ser verdadeiro.** É informação de fama, não de áudio: deve vir de dados reais do artista. A API aceita os três números
  opcionalmente; a UI deixa ver a mesma faixa nas duas modalidades.
- **Lotes mistos têm escalas diferentes.** O score com histórico é muito mais espalhado que o só-áudio; num CSV que mistura as duas, as faixas com
  histórico tendem a ocupar o topo e o fundo. A tela de ranking avisa e oferece "Ignorar o histórico do artista" para comparar em pé de igualdade.
- **O percentil de treino foi calculado antes do split** (com as linhas de teste juntas no ranking por gênero): efeito de segunda ordem, aceito e documentado.

## Consequências
`PredictionResponse` ganha `mode` e `expected_popularity`; o lote devolve `mode` por faixa e um threshold por modalidade (API 1.2.0).
O drift de predição compara só a modalidade só-áudio (a das demos); misturar as duas geraria falso alarme. O gate exige que **nenhuma**
das duas modalidades piore.
