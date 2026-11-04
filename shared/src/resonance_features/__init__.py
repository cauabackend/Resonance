"""Engenharia de features compartilhada: o MESMO código no treino e na inferência (sem train/serve skew).

Entrada: linhas com as 9 features de áudio, `track_genre` e, opcionalmente, mais 5 features de áudio e o histórico do
artista (3 números). Saída: o DataFrame que o modelo consome (`MODEL_COLUMNS` + `track_genre`).

Duas modalidades convivem no mesmo modelo (o XGBoost trata NaN nativamente):
- **só áudio** (demo de artista sem histórico): as colunas de artista chegam vazias;
- **com histórico do artista** (artista que já tem faixas publicadas): n de faixas, popularidade média e fração de hits.
"""

import numpy as np
import pandas as pd

GENRE = "track_genre"
AUDIO = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "speechiness", "acousticness", "instrumentalness", "duration_ms",
]  # fmt: skip
EXTRA = ["liveness", "key", "mode", "time_signature", "explicit"]
ARTIST = ["artist_n_tracks", "artist_mean_popularity", "artist_hit_share"]
Z_COLS = [f"{f}_z" for f in AUDIO]
DERIVED = ["energy_x_danceability", "artist_pct_in_genre"]
MODEL_COLUMNS = [*AUDIO, *EXTRA, *Z_COLS, DERIVED[0], *ARTIST, DERIVED[1]]
QUANTILE_GRID = np.linspace(0, 100, 101)

# Fator de explicação (SHAP) -> colunas do modelo que ele agrega
FACTOR_COLUMNS: dict[str, list[str]] = {
    **{f: [f, f"{f}_z"] for f in AUDIO},
    **{f: [f] for f in EXTRA},
    "energy_x_danceability": ["energy_x_danceability"],
    "artist_context": [*ARTIST, "artist_pct_in_genre"],
}
# Coluna do dado cru que representa o valor de cada fator na explicação
FACTOR_VALUE = {**{f: f for f in [*AUDIO, *EXTRA]}, "energy_x_danceability": "energy_x_danceability",
                "artist_context": "artist_mean_popularity", GENRE: GENRE}  # fmt: skip


def fit_stats(df: pd.DataFrame) -> dict:
    """Estatísticas por gênero (média/desvio das features e quantis da popularidade), só do conjunto de treino."""
    by = df.groupby(GENRE)
    return {
        "genre_mean": by[AUDIO].mean().to_dict("index"),
        "genre_std": by[AUDIO].std().to_dict("index"),
        "quantiles": {g: np.percentile(d["popularity"], QUANTILE_GRID).tolist() for g, d in by},
        "global_quantiles": np.percentile(df["popularity"], QUANTILE_GRID).tolist(),
    }


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").astype("float64")  # bool vira 0/1, None vira NaN


def pct_in_genre(popularity: np.ndarray, genres: np.ndarray, stats: dict) -> np.ndarray:
    """Percentil (0-100) que uma popularidade ocuparia dentro de cada gênero (inverso da tabela de quantis)."""
    out = np.full(len(popularity), np.nan)
    for g in np.unique(genres):
        q = stats["quantiles"].get(g, stats["global_quantiles"])
        m = (genres == g) & ~np.isnan(popularity)
        out[m] = np.interp(popularity[m], q, QUANTILE_GRID)
    return out


def expected_popularity(scores: np.ndarray, genres: np.ndarray, stats: dict) -> np.ndarray:
    """Converte o percentile_score previsto na popularidade esperada NAQUELE gênero (quantil do gênero no percentil).

    É isso que devolve a informação de gênero ao badge: percentil 80 em pop é popularidade ~60; em grindcore, ~20.
    """
    scores, genres = np.asarray(scores, dtype=float), np.asarray(genres)
    out = np.empty(len(scores))
    for g in np.unique(genres):
        q = stats["quantiles"].get(g, stats["global_quantiles"])
        m = genres == g
        out[m] = np.interp(scores[m], QUANTILE_GRID, q)
    return out


def _tables(stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Médias e desvios por gênero como DataFrames, guardados no próprio `stats` (sem cache global por id())."""
    if "_tables" not in stats:
        stats["_tables"] = (
            pd.DataFrame.from_dict(stats["genre_mean"], orient="index")[AUDIO],
            pd.DataFrame.from_dict(stats["genre_std"], orient="index")[AUDIO],
        )
    return stats["_tables"]


def engineer(df: pd.DataFrame, stats: dict, use_artist_context: bool = True) -> pd.DataFrame:
    """Monta as colunas do modelo. Colunas ausentes viram NaN; `use_artist_context=False` força a modalidade só-áudio."""
    n = len(df)
    cols: dict[str, np.ndarray] = {}
    for c in [*AUDIO, *EXTRA, *ARTIST]:
        cols[c] = _num(df, c).to_numpy() if c in df else np.full(n, np.nan)
    genres = df[GENRE].to_numpy()
    mean, std = _tables(stats)
    sd = np.where(std.reindex(genres).to_numpy() == 0, np.nan, std.reindex(genres).to_numpy())
    z = (np.column_stack([cols[f] for f in AUDIO]) - mean.reindex(genres).to_numpy()) / sd
    for i, f in enumerate(AUDIO):
        cols[f"{f}_z"] = z[:, i]
    cols["energy_x_danceability"] = cols["energy"] * cols["danceability"]
    if not use_artist_context:
        for c in ARTIST:
            cols[c] = np.full(n, np.nan)
    cols["artist_pct_in_genre"] = pct_in_genre(cols["artist_mean_popularity"], genres, stats)
    out = pd.DataFrame({c: cols[c] for c in MODEL_COLUMNS}, index=df.index)
    out[GENRE] = genres
    return out


def factor_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """Índices das colunas transformadas (saída do ColumnTransformer) que formam cada fator de explicação."""
    groups: dict[str, list[int]] = {f: [] for f in FACTOR_COLUMNS} | {GENRE: []}
    for i, name in enumerate(feature_names):
        kind, _, col = name.partition("__")
        if kind == "genre":
            groups[GENRE].append(i)
            continue
        for factor, cols in FACTOR_COLUMNS.items():
            if col in cols:
                groups[factor].append(i)
    return {k: v for k, v in groups.items() if v}


def aggregate_shap(sv: np.ndarray, groups: dict[str, list[int]]) -> tuple[list[str], np.ndarray]:
    """Soma os SHAP das colunas de cada fator (o efeito de `danceability` e de `danceability_z` é um só)."""
    names = list(groups)
    return names, np.column_stack([sv[:, idx].sum(axis=1) for idx in groups.values()])
