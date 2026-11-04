import numpy as np
import pandas as pd
from resonance_features import ARTIST, engineer, expected_popularity

from app.schemas.prediction import ModelMetadata, PredictionResponse
from app.schemas.track import TrackFeatures
from app.services.explain import ExplainService


class InferenceService:
    """Tempo real e batch compartilham o mesmo modelo; muda só a forma de entrada/saída.

    Duas modalidades no mesmo modelo: **artist_aware** (a faixa traz o histórico do artista) e **audio_only** (não traz,
    ou `use_artist_context=False`). Cada uma tem o seu calibrador e o seu threshold de badge.
    """

    def __init__(self, bundle: dict, sha256: str = ""):
        self.bundle = bundle
        self.sha256 = sha256
        self.version: str = bundle["version"]
        self.stats: dict = bundle["stats"]
        self.explainer = ExplainService(bundle)

    def _frame(self, tracks: list[TrackFeatures], use_artist: bool = True) -> pd.DataFrame:
        df = pd.DataFrame([t.model_dump() for t in tracks])
        if not use_artist:
            df[ARTIST] = None
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        return engineer(df, self.stats)

    @staticmethod
    def modes(df: pd.DataFrame) -> np.ndarray:
        """artist_aware se a linha traz a popularidade média do artista; senão audio_only."""
        return np.where(df["artist_mean_popularity"].notna(), "artist_aware", "audio_only")

    def score(self, tracks: list[TrackFeatures], use_artist: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(percentile_score 0-100, modalidade por faixa). Vetorizado: uma chamada ao XGBoost para o lote inteiro."""
        df = self._frame(tracks, use_artist)
        e = self._engineer(df)
        raw = self.bundle["model"].predict(self.bundle["preprocessor"].transform(e))
        return np.clip(raw, 0, 100), self.modes(e)

    def calibrate(self, scores: np.ndarray, genres: np.ndarray, modes: np.ndarray) -> tuple[np.ndarray, ...]:
        """(popularidade esperada no gênero, P(hit), threshold) por faixa, com o calibrador da modalidade."""
        ep = expected_popularity(scores, genres, self.stats)
        prob, thr = np.empty(len(ep)), np.empty(len(ep))
        for mode in ("artist_aware", "audio_only"):
            m = modes == mode
            if m.any():
                prob[m] = self.bundle["calibrators"][mode].predict(ep[m])
                thr[m] = self.bundle["thresholds"][mode]
        return ep, prob, thr

    def score_and_calibrate(self, tracks: list[TrackFeatures], use_artist: bool = True) -> dict[str, np.ndarray]:
        scores, modes = self.score(tracks, use_artist)
        genres = np.array([t.track_genre for t in tracks])
        ep, prob, thr = self.calibrate(scores, genres, modes)
        return {"score": scores, "mode": modes, "expected_popularity": ep, "probability": prob, "threshold": thr,
                "is_hit": prob >= thr}  # fmt: skip

    def predict(self, track: TrackFeatures, use_artist: bool = True) -> PredictionResponse:
        r = self.score_and_calibrate([track], use_artist)
        factors = self.explainer.explain(self._engineer(self._frame([track], use_artist)))
        return PredictionResponse(
            percentile_score=float(r["score"][0]),
            mode=str(r["mode"][0]),  # type: ignore[arg-type]
            expected_popularity=float(r["expected_popularity"][0]),
            is_hit=bool(r["is_hit"][0]),
            probability=float(r["probability"][0]),
            threshold=float(r["threshold"][0]),
            model_version=self.version,
            top_factors=factors[:3],
            factors=factors,
        )

    def metadata(self) -> ModelMetadata:
        m, b = self.bundle["metrics"], self.bundle["baseline"]
        return ModelMetadata(
            version=self.version,
            created_at=self.bundle.get("created_at"),
            threshold=self.bundle["thresholds"]["audio_only"],
            metrics={k: v for k, v in m.items() if isinstance(v, float)},
            baseline_ridge_ndcg_at_k=b["ridge"]["ndcg_at_k"],
            baseline_ridge_spearman=b["ridge"]["spearman"],
            baseline_artist_mean_ndcg_at_k=b["artist_mean"]["ndcg_at_k"],
            baseline_artist_mean_spearman=b["artist_mean"]["spearman"],
            dataset_md5=self.bundle.get("dataset_md5", "unknown"),
            hit_means=self.bundle["hit_means"],
            known_genres=self.bundle["known_genres"],
        )
