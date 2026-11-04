from pydantic import BaseModel, ConfigDict, Field


class TrackFeatures(BaseModel):
    """As 10 entradas obrigatórias bastam (modalidade só-áudio). Os campos opcionais melhoram a predição quando existem.

    - `liveness`, `key`, `mode`, `time_signature`, `explicit`: demais features de áudio/letra do Spotify.
    - `artist_*`: histórico do artista (modalidade com histórico), das OUTRAS faixas já publicadas dele.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    danceability: float = Field(ge=0, le=1)
    energy: float = Field(ge=0, le=1)
    valence: float = Field(ge=0, le=1)
    tempo: float = Field(ge=30, le=300)
    loudness: float = Field(ge=-60, le=5)
    speechiness: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)
    duration_ms: int = Field(gt=0, le=900_000)
    track_genre: str = Field(min_length=1, max_length=64)  # entra no modelo via encoding, não fica só na EDA

    liveness: float | None = Field(default=None, ge=0, le=1)
    key: int | None = Field(default=None, ge=-1, le=11)
    mode: int | None = Field(default=None, ge=0, le=1)
    time_signature: int | None = Field(default=None, ge=0, le=7)
    explicit: bool | None = None

    artist_n_tracks: int | None = Field(default=None, ge=0, le=10_000, description="Outras faixas do artista")
    artist_mean_popularity: float | None = Field(
        default=None, ge=0, le=100, description="Popularidade média (0-100) das outras faixas do artista"
    )
    artist_hit_share: float | None = Field(
        default=None, ge=0, le=1, description="Fração das outras faixas do artista com popularidade ≥ 60"
    )


class TrackOut(BaseModel):
    track_id: str
    name: str
    artist: str
    popularity: int
    features: TrackFeatures
