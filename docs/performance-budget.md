# Orçamento de performance

Restrições de engenharia tratadas como requisito, com o teste que as protege.

| Métrica | Orçamento | Medido (dev, CPU, 1 worker) | Onde é verificado |
|---|---|---|---|
| Latência p95 de `POST /v1/predict` (com SHAP) | **< 200 ms** | p95 ≈ 22 ms (sequencial); p95 ≈ 173 ms com 8 clientes simultâneos | `api/tests/integration/test_api.py::test_latency_budget_p95_under_200ms` (CI) |
| Throughput de `POST /v1/predict/batch` (500 faixas) | > 5.000 faixas/s | ~13.800 faixas/s | `python api/scripts/benchmark.py` (manual) |
| Lote máximo | 1.000 faixas (413 acima) | n/a | testes de integração |
| Rate limit | 120 req/min por IP | n/a | `test_rate_limit_returns_429_but_not_for_health` |
| Memória da API | < 500 MB RSS | **não medida** | só documentado; medir com `docker stats` |

O orçamento de latência é verificado no CI com o `TestClient` (sem rede), com um modelo pequeno; ele pega regressões grosseiras
(por exemplo, voltar a criar o `TreeExplainer` a cada requisição), não substitui o teste de carga real.

## Teste de carga
`python api/scripts/benchmark.py --concurrency 8` (threads + httpx, sem ferramenta externa) imprime p50/p95/p99 com clientes
simultâneos. **Suba a API com `RATE_LIMIT_PER_MINUTE=0`**, senão o limitador devolve 429 ao próprio teste. Locust/k6 ficam
como evolução: seriam dependências novas para o mesmo número.
