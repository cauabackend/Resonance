"""Latência do /predict (p50/p95) vs throughput do /predict/batch, contra uma API em execução.

Uso: python scripts/benchmark.py [--url http://localhost:8000] [--n 200] [--batch 500] [--concurrency 8]
Suba a API com RATE_LIMIT_PER_MINUTE=0, senão o limitador devolve 429 ao próprio teste.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

FEATURES = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "speechiness", "acousticness", "instrumentalness", "duration_ms", "track_genre",
]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--data", default="../data/processed/tracks.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=8, help="clientes simultâneos no teste de carga")
    a = ap.parse_args()

    rows = pd.read_csv(a.data, usecols=FEATURES).sample(a.batch, random_state=0).to_dict("records")
    with httpx.Client(base_url=a.url, timeout=60) as c:
        for r in rows[:10]:  # aquecimento
            c.post("/v1/predict", json={"features": r})

        lat = []
        for r in rows[: a.n] if a.n <= len(rows) else (rows * a.n)[: a.n]:
            t = time.perf_counter()
            c.post("/v1/predict", json={"features": r}).raise_for_status()
            lat.append((time.perf_counter() - t) * 1000)
        q = statistics.quantiles(lat, n=100)
        single_tps = 1000 / statistics.mean(lat)

        t = time.perf_counter()
        c.post("/v1/predict/batch", json={"tracks": rows}).raise_for_status()
        batch_s = time.perf_counter() - t

    # Teste de carga: N clientes em paralelo contra /v1/predict (sem dependência nova: threads + httpx).
    def worker(chunk):
        out = []
        with httpx.Client(base_url=a.url, timeout=60) as c2:
            for r in chunk:
                t0 = time.perf_counter()
                c2.post("/v1/predict", json={"features": r, "source": "catalog"}).raise_for_status()
                out.append((time.perf_counter() - t0) * 1000)
        return out

    chunks = [rows[i :: a.concurrency] * 4 for i in range(a.concurrency)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        load = [x for part in ex.map(worker, chunks) for x in part]
    load_s = time.perf_counter() - t0
    lq = statistics.quantiles(load, n=100)

    print("| Endpoint | Resultado |\n|---|---|")
    print(f"| POST /v1/predict (n={len(lat)}) | p50 {q[49]:.1f} ms · p95 {q[94]:.1f} ms · ~{single_tps:.0f} tracks/s |")
    print(
        f"| POST /v1/predict, {a.concurrency} clientes simultâneos (n={len(load)}) | p50 {lq[49]:.0f} ms · p95 {lq[94]:.0f} ms · p99 {lq[98]:.0f} ms · {len(load) / load_s:.0f} req/s |"
    )
    print(
        f"| POST /v1/predict/batch ({len(rows)} tracks) | {batch_s * 1000:.0f} ms · {len(rows) / batch_s:.0f} tracks/s |"
    )


if __name__ == "__main__":
    main()
