"""Contrato dos dados. Fonte única: o DATA_DICTIONARY.md é gerado daqui."""

import pandera.pandas as pa
from pandera.pandas import Check, Column

PCT = "popularity_percentile_within_genre"

_audio = {
    **{
        c: Column(float, Check.in_range(0, 1), coerce=True)
        for c in ("danceability", "energy", "valence", "speechiness", "acousticness", "instrumentalness")
    },
    "loudness": Column(float, Check.in_range(-60, 5), coerce=True),
}


def _schema(tempo: Column, duration: Column, text_nullable: bool, targets: bool = False) -> pa.DataFrameSchema:
    cols = {
        "name": Column(str, nullable=text_nullable),
        "artist": Column(str, nullable=text_nullable),
        "popularity": Column(int, Check.in_range(0, 100), coerce=True),
        "track_genre": Column(str, nullable=False),
        "liveness": Column(float, Check.in_range(0, 1), coerce=True),
        "key": Column(int, Check.in_range(-1, 11), coerce=True),
        "mode": Column(int, Check.isin([0, 1]), coerce=True),
        "time_signature": Column(int, Check.in_range(0, 7), coerce=True),
        "explicit": Column(bool, coerce=True),
        "track_id": Column(str, nullable=True, required=False),
        **_audio,
        "tempo": tempo,
        "duration_ms": duration,
    }
    if targets:
        cols[PCT] = Column(float, Check.in_range(0, 100), coerce=True)
        cols["is_hit"] = Column(int, Check.isin([0, 1]), coerce=True)
        # Histórico do artista (leave-one-out): NaN quando o artista só tem esta faixa.
        cols["artist_n_tracks"] = Column(float, Check.ge(0), nullable=True, coerce=True)
        cols["artist_mean_popularity"] = Column(float, Check.in_range(0, 100), nullable=True, coerce=True)
        cols["artist_hit_share"] = Column(float, Check.in_range(0, 1), nullable=True, coerce=True)
    return pa.DataFrameSchema(cols, strict=False)


_TEMPO = Column(float, Check.ge(30), coerce=True)
_DURATION = Column(int, Check.in_range(1, 900_000), coerce=True)

# Raw aceita os defeitos conhecidos (tempo/duração 0, nome nulo); o preprocess os remove.
RAW_SCHEMA = _schema(Column(float, Check.ge(0), coerce=True), Column(int, Check.ge(0), coerce=True), True)
CLEAN_SCHEMA = _schema(_TEMPO, _DURATION, False)
# Saída do estágio `target`: clean + alvo de treino (percentil no gênero) + rótulo histórico do badge.
TRACKS_SCHEMA = _schema(_TEMPO, _DURATION, False, targets=True)

# O pandera não carrega nota de qualidade por coluna.
NOTES = {
    "tempo": "valores próximos de 0 indicam falha de detecção de BPM; tratados como outlier (< 30)",
    "loudness": "correlacionada com energy (VIF em reports/analysis.json)",
    "duration_ms": "faixas > 15 min são provavelmente podcast/mix; filtradas",
    "track_genre": "categorias raras (< 100 exemplos) agrupadas pelo encoder",
    "popularity": "recency-weighted, snapshot estático; não comparável entre anos (sem release_date)",
    "name": "dedup por track_id; sem id, name+artist+duration_ms",
    "artist": "primeiro artista listado; também é o grupo dos splits (sem vazamento entre artistas)",
    "track_id": "id do Spotify (opcional)",
    PCT: "alvo real de treino: percentil (0-100) da popularidade dentro do próprio gênero; recalculado em target.py",
    "artist_n_tracks": "OUTRAS faixas do mesmo artista no dataset (leave-one-out: nunca inclui a própria linha)",
    "artist_mean_popularity": "popularidade média das OUTRAS faixas do artista; NaN se o artista só tem esta faixa",
    "artist_hit_share": "fração das OUTRAS faixas do artista com popularity ≥ 60",
    "liveness": "probabilidade de ser gravação ao vivo (Spotify)",
    "key": "tom (0-11, -1 = não detectado)",
    "mode": "0 = menor, 1 = maior",
    "time_signature": "compasso estimado (0-7)",
    "explicit": "conteúdo explícito (sinal de letra, disponível para o demo)",
    "is_hit": "rótulo histórico (popularity ≥ 60); NÃO é alvo de treino, só calibra a probabilidade do badge",
}


def _fmt(stats: dict) -> str:
    lo, hi = stats.get("min_value"), stats.get("max_value")
    if lo is not None and hi is not None:
        return f"{lo:g}–{hi:g}"
    if lo is not None:
        return f"≥ {lo:g}"
    return "{" + ", ".join(map(str, stats.get("allowed_values", []))) + "}"


def data_dictionary_md() -> str:
    rows = ["| Coluna | Tipo | Range válido | Nota de qualidade |", "|---|---|---|---|"]
    for name, col in TRACKS_SCHEMA.columns.items():
        rng = ", ".join(_fmt(c.statistics) for c in col.checks) or "—"
        rows.append(f"| `{name}` | {col.dtype} | {rng} | {NOTES.get(name, 'sem outliers conhecidos')} |")
    return (
        "# Dicionário de dados\n\n"
        "Gerado de `pipeline/src/resonance_pipeline/schema.py` (`dvc repro validate`). "
        "Não editar à mão.\n\n" + "\n".join(rows) + "\n"
    )
