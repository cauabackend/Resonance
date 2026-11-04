from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.track import TrackFeatures


class ShapFactor(BaseModel):
    feature: str
    value: float | str
    shap_value: float = Field(description="Contribuição, em pontos de percentil, para o percentile_score previsto")
    direction: Literal["increases", "decreases"]


class PredictionRequest(BaseModel):
    features: TrackFeatures
    # "catalog": faixa do próprio dataset de treino (navegação na UI). Não entra no log de drift.
    source: Literal["user", "catalog"] = "user"
    # False ignora o histórico do artista mesmo que enviado: "como esta faixa seria avaliada se o artista fosse novo?"
    use_artist_context: bool = True


Mode = Literal["artist_aware", "audio_only"]


class PredictionResponse(BaseModel):
    percentile_score: float = Field(
        ge=0, le=100, description="Percentil previsto de popularidade dentro do gênero (0-100): o alvo real do modelo"
    )
    mode: Mode = Field(description="artist_aware: usou o histórico do artista; audio_only: só as features da faixa")
    expected_popularity: float = Field(
        ge=0, le=100, description="Popularidade (0-100) esperada NO GÊNERO para esse percentil previsto"
    )
    is_hit: bool = Field(description="Badge: probability >= threshold (threshold calibrado por custo de negócio)")
    probability: float = Field(ge=0, le=1, description="P(hit histórico), isotônica sobre o percentile_score")
    threshold: float
    model_version: str
    top_factors: list[ShapFactor]
    factors: list[ShapFactor] = Field(description="Todos os fatores, do maior ao menor impacto")


class BatchItem(BaseModel):
    input_index: int = Field(description="Posição da faixa na lista enviada")
    rank: int = Field(description="Posição no lote, 1 = maior percentile_score")
    mode: Mode
    percentile_score: float
    expected_popularity: float
    probability: float
    is_hit: bool


class BatchRequest(BaseModel):
    tracks: list[TrackFeatures] = Field(min_length=1, max_length=1000)  # teto duro; max_batch_size pode reduzir
    use_artist_context: bool = True


class BatchPredictionResponse(BaseModel):
    model_version: str
    thresholds: dict[str, float] = Field(description="Threshold do badge por modalidade (artist_aware, audio_only)")
    results: list[BatchItem] = Field(description="Já ordenado por percentile_score decrescente")
    rank: list[int] = Field(description="rank[i] = posição da i-ésima faixa enviada dentro do lote")


class ModelMetadata(BaseModel):
    version: str
    created_at: str | None = None
    threshold: float
    metrics: dict[str, float]
    baseline_ridge_ndcg_at_k: float = Field(description="Piso da modalidade só-áudio (regressão linear)")
    baseline_ridge_spearman: float
    baseline_artist_mean_ndcg_at_k: float = Field(description="Piso da modalidade com histórico (média do artista)")
    baseline_artist_mean_spearman: float
    dataset_md5: str
    hit_means: dict[str, float]
    known_genres: list[str]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None
    catalog_loaded: bool


class DeepHealth(BaseModel):
    status: Literal["ok", "mismatch", "degraded"]
    model_version: str | None
    loaded_sha256: str | None
    registry_sha256: str | None = Field(description="Hash do artefato que o registry serve AGORA (alias/versão atual)")
    detail: str
