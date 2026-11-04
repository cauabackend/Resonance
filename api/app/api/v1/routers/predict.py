import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_inference
from app.schemas.prediction import (
    BatchItem,
    BatchPredictionResponse,
    BatchRequest,
    PredictionRequest,
    PredictionResponse,
)
from app.services.inference import InferenceService

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
def predict(req: PredictionRequest, request: Request, svc: InferenceService = Depends(get_inference)):
    """Uma música, resposta interativa (com explicação SHAP em pontos de percentil)."""
    out = svc.predict(req.features, req.use_artist_context)
    if req.source == "user":  # navegar no catálogo de treino não pode poluir o monitoramento de drift
        request.app.state.prediction_logger.log(
            request.state.request_id,
            svc.version,
            [(req.features.model_dump(), out.percentile_score, out.probability, out.is_hit, out.mode)],
        )
    return out


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchRequest, request: Request, svc: InferenceService = Depends(get_inference)):
    """Lote vetorizado (ex.: 500 demos), devolvido ORDENADO por percentile_score, sem explicação SHAP."""
    limit = request.app.state.settings.max_batch_size
    if len(req.tracks) > limit:
        raise HTTPException(413, f"Máximo de {limit} músicas por lote")
    r = svc.score_and_calibrate(req.tracks, req.use_artist_context)
    scores = r["score"]
    order = np.argsort(-scores, kind="stable")  # maior primeiro; empates mantêm a ordem de entrada
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    results = [
        BatchItem(
            input_index=int(i),
            rank=int(rank[i]),
            mode=str(r["mode"][i]),  # type: ignore[arg-type]
            percentile_score=float(scores[i]),
            expected_popularity=float(r["expected_popularity"][i]),
            probability=float(r["probability"][i]),
            is_hit=bool(r["is_hit"][i]),
        )
        for i in order
    ]
    request.app.state.prediction_logger.log(
        request.state.request_id,
        svc.version,
        [
            (t.model_dump(), float(scores[i]), float(r["probability"][i]), bool(r["is_hit"][i]), str(r["mode"][i]))
            for i, t in enumerate(req.tracks)
        ],
    )
    return BatchPredictionResponse(
        model_version=svc.version,
        thresholds={m: svc.bundle["thresholds"][m] for m in ("artist_aware", "audio_only")},
        results=results,
        rank=[int(x) for x in rank],
    )
