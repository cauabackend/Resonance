# 0001. FastAPI como camada de serving separada do front

**Status:** aceito

## Contexto
O app original (Streamlit) carregava modelo e CSV do disco, calculava SHAP e desenhava gráficos no mesmo arquivo.
Não havia fronteira para testar, escalar ou trocar a interface.

## Decisão
Serving em FastAPI (`api/`), com camadas router → service → registry loader. Routers só validam entrada/saída
(Pydantic) e chamam um service. O front React consome só o contrato OpenAPI (`/v1`), e os tipos TypeScript são
**gerados** dele (`openapi-typescript`), então divergência vira erro de build.

## Alternativas
- **Flask:** sem validação/OpenAPI nativos; o contrato tipado é justamente o ponto.
- **Manter Streamlit + API:** duas UIs para manter; o pedido já exigia React.

## Consequências
Cada resposta carrega `model_version` (correlação com drift/incidentes). `/v1/predict` (tempo real, com SHAP) e
`/v1/predict/batch` (vetorizado, sem SHAP) compartilham o mesmo `InferenceService`. Medido localmente:
p50 ≈ 21 ms / p95 ≈ 22 ms no individual (com SHAP e a engenharia de features); ~13,8 mil faixas/s em lote de 500.
