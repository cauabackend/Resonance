# Resonance

## O problema

Uma gravadora independente com orçamento apertado precisa escolher em quais artistas novos apostar. Contratar A&R experiente custa caro, e intuição sozinha não escala. A pergunta que esse projeto tenta responder: olhando só pras características de áudio de uma música, dá pra ter alguma ideia se ela vai estourar ou não?

O modelo não substitui ouvido humano. Ele filtra. Em vez de ouvir 500 demos, a gravadora pode usar o app pra separar as 50 que valem uma escuta mais atenta.

## Stack

| Ferramenta | O que faz aqui |
|---|---|
| Python 3.12+ | Linguagem do projeto inteiro |
| Pandas | Limpeza e manipulação dos dados |
| Scikit-learn | Split treino/teste, métricas, Random Forest |
| XGBoost | Modelo principal de classificação |
| SHAP | Explica por que o modelo decide o que decide |
| Plotly | Gráficos interativos (todos os do projeto) |
| Streamlit | Interface web onde o usuário testa músicas |
| JupyterLab | Ambiente dos notebooks de análise |

## Como o projeto está organizado

| Notebook | O que responde |
|---|---|
| `01_coleta.ipynb` | Documentação do processo de coleta de dados |
| `02_eda.ipynb` | O que diferencia um hit de uma música que não estourou? |
| `03_generos.ipynb` | Como as features de áudio variam entre gêneros? |
| `04_modelagem.ipynb` | Random Forest ou XGBoost — qual funciona melhor? |
| `05_interpretabilidade.ipynb` | Quais features mais pesam na decisão do modelo? |
| `06_analise_de_erros.ipynb` | Onde e por que o modelo erra? |

## Dataset

Obtido do Kaggle (Spotify Tracks Dataset), com aproximadamente 114 mil tracks cobrindo mais de 100 gêneros musicais. Após limpeza, o dataset final tem cerca de 81 mil músicas.

Features de áudio: danceability, energy, valence, tempo, loudness, speechiness, acousticness, instrumentalness, duration_ms. Target: `is_hit` (popularity >= 60 na API do Spotify).

## Resultados

| Métrica | Random Forest | XGBoost |
|---|---|---|
| Accuracy | XX.X% | XX.X% |
| Precision | XX.X% | XX.X% |
| Recall | XX.X% | XX.X% |
| F1 | XX.X% | XX.X% |
| ROC-AUC | XX.X% | XX.X% |

*Preencha com os valores que aparecerem no notebook 04 após rodar.*

## O que o modelo aprendeu

- Músicas dançáveis e com produção de volume alto têm mais chance de estourar. Faixas acústicas e instrumentais ficam em desvantagem nos charts.
- Ser triste ou alegre não faz diferença. BPM também não. O público do Spotify não tem preferência clara por andamento ou humor da música.
- Duração importa. Faixas mais curtas (até 3:30) performam melhor — playlists de streaming favorecem músicas que não perdem o ouvinte.

## Diferenças entre gêneros

- O perfil de hit muda entre gêneros. Um hit de hip-hop tem speechiness alta e danceability forte. Um hit de rock tem energy e loudness no topo mas speechiness baixa.
- Gêneros de nicho (grindcore, black metal, study music) quase não têm hits no dataset. O modelo tem menos material pra aprender ali e erra mais.

## Limitações

O modelo usa só features de áudio. Não sabe se o artista viralizou no TikTok, se a música entrou em novela, se teve feat com alguém grande, nem se a gravadora investiu em marketing. Hits que estouraram por contexto externo vão escapar. Na faixa de popularidade entre 45 e 70, as predições ficam menos confiáveis. O app funciona melhor como filtro inicial do que como decisão final.

O dataset veio do Kaggle e não tem informação de ano de lançamento, o que impediu a análise temporal. A coleta via API do Spotify não foi possível por restrições recentes (fevereiro de 2026) no endpoint de playlist tracks e audio features pra apps em modo de desenvolvimento.

## Como rodar

```bash
git clone https://github.com/seu-usuario/music-hit-predictor.git
cd music-hit-predictor

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python src/preprocess.py
python src/train.py

streamlit run app/streamlit_app.py
```

## Demo

Acesse o app em: [link do Streamlit Community Cloud]