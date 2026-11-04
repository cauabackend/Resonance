"""Alvo de treino: percentil de popularidade dentro do gênero (substitui o corte global is_hit >= 60)."""

import numpy as np
import pandas as pd

from .common import GENRE, TARGET, load_params, repo_path
from .schema import PCT, TRACKS_SCHEMA


def add_artist_context(df: pd.DataFrame, hit_threshold: int) -> pd.DataFrame:
    """Histórico do artista, leave-one-out: estatísticas das OUTRAS faixas do artista (nunca a própria linha).

    Sem leak do alvo da própria faixa. Como o split é por artista, treino e teste não compartilham artistas; no teste
    o histórico vem das outras faixas do mesmo artista, como seria na vida real (faixas já publicadas dele).
    """
    g = df.groupby("artist")["popularity"]
    n_others = g.transform("count") - 1
    others = n_others.replace(0, np.nan)
    hits = (df["popularity"] >= hit_threshold).astype(int)
    out = df.copy()
    out["artist_n_tracks"] = n_others.astype(float)
    out["artist_mean_popularity"] = (g.transform("sum") - df["popularity"]) / others
    out["artist_hit_share"] = (hits.groupby(df["artist"]).transform("sum") - hits) / others
    return out


def add_targets(df: pd.DataFrame, hit_threshold: int) -> pd.DataFrame:
    """Percentil 0-100 por gênero (empates = média dos ranks) + rótulo histórico do badge."""
    df = df.copy()
    df[PCT] = df.groupby(GENRE)["popularity"].rank(pct=True, method="average") * 100
    df[TARGET] = (df["popularity"] >= hit_threshold).astype(int)
    df = add_artist_context(df, hit_threshold)
    return TRACKS_SCHEMA.validate(df, lazy=True)


def main() -> None:
    p = load_params()["data"]
    df = add_targets(pd.read_csv(repo_path(p["clean"])), p["hit_threshold"])
    out = repo_path(p["processed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(
        f"Alvo calculado: {len(df)} faixas em {df[GENRE].nunique()} gêneros | hits históricos: {df[TARGET].mean():.1%}"
    )


if __name__ == "__main__":
    main()
