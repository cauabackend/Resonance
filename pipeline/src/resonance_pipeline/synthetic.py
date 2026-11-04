"""Gerador de dataset sintético: quem clonar o repo roda `dvc repro` sem baixar as ~114 mil faixas do Kaggle.

    python -m resonance_pipeline.synthetic --n 6000 --out data/raw/dataset.csv

O sinal é fabricado (popularidade = base do gênero + efeito do artista + danceability/loudness + ruído), então os
resultados do pipeline NÃO valem como resultado real; servem para testar o pipeline e a API de ponta a ponta.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["pop", "rock", "hip-hop", "jazz", "metal", "ambient", "samba", "grindcore", "study", "k-pop"]


def make_dataset(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genre = rng.choice(GENRES, n, p=np.array([0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.08, 0.05, 0.04, 0.05]))
    n_artists = max(n // 12, 10)
    artist = rng.integers(0, n_artists, n)
    artist_effect = rng.normal(0, 12, n_artists)  # a fama do artista explica boa parte da popularidade
    genre_base = dict(zip(GENRES, rng.uniform(10, 55, len(GENRES))))
    df = pd.DataFrame(
        {
            "track_id": [f"syn{i:06d}" for i in range(n)],
            "name": [f"Faixa sintética {i}" for i in range(n)],
            "artist": [f"Artista {a}" for a in artist],
            "track_genre": genre,
            "danceability": rng.beta(5, 3, n),
            "energy": rng.beta(4, 3, n),
            "valence": rng.beta(3, 3, n),
            "speechiness": rng.beta(1.2, 12, n),
            "acousticness": rng.beta(1.2, 2.5, n),
            "instrumentalness": rng.beta(0.4, 3, n),
            "liveness": rng.beta(1.2, 6, n),
            "key": rng.integers(0, 12, n),
            "mode": rng.integers(0, 2, n),
            "time_signature": rng.choice([3, 4, 5], n, p=[0.1, 0.85, 0.05]),
            "explicit": rng.random(n) < 0.15,
            "loudness": np.clip(rng.normal(-8, 4, n), -40, 2),
            "tempo": np.clip(rng.normal(120, 28, n), 40, 220),
            "duration_ms": np.clip(rng.normal(215_000, 55_000, n), 60_000, 600_000).astype(int),
        }
    )
    pop = (
        np.array([genre_base[g] for g in genre])
        + artist_effect[artist]
        + 18 * (df["danceability"] - 0.6)
        + 0.8 * (df["loudness"] + 8)
        + rng.normal(0, 9, n)
    )
    df["popularity"] = np.clip(pop, 0, 100).round().astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/dataset.csv")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_dataset(a.n, a.seed).to_csv(out, index=False)
    print(f"{a.n} faixas sintéticas em {out}")


if __name__ == "__main__":
    main()
