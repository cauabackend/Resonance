import time

from fastapi import APIRouter, Depends, Request, Response

from app.api.deps import get_inference
from app.models_registry.loader import resolve_bundle_file, sha256_of
from app.schemas.prediction import DeepHealth, Health, ModelMetadata
from app.services.inference import InferenceService

router = APIRouter(tags=["model"])

DEEP_TTL = 60.0  # s: o health profundo baixa o artefato do registry; sem cache seria um amplificador de carga


@router.get("/model/metadata", response_model=ModelMetadata)
def metadata(svc: InferenceService = Depends(get_inference)):
    return svc.metadata()


@router.get("/health", response_model=Health)
def health(request: Request, response: Response):
    inference, catalog = request.app.state.inference, request.app.state.catalog
    ok = inference is not None and catalog is not None
    response.status_code = 200 if ok else 503
    return Health(
        status="ok" if ok else "degraded",
        model_version=inference.version if inference else None,
        catalog_loaded=catalog is not None,
    )


@router.get("/health/deep", response_model=DeepHealth)
def health_deep(request: Request, response: Response):
    """Valida que o modelo em memória é o artefato que o registry serve agora (não só que o processo está de pé)."""
    svc = request.app.state.inference
    if svc is None:
        response.status_code = 503
        return DeepHealth(status="degraded", model_version=None, loaded_sha256=None, registry_sha256=None,
                          detail="modelo não carregado")  # fmt: skip
    cfg = request.app.state.settings
    key = (cfg.model_registry_uri, cfg.model_version_pin)
    cached = getattr(request.app.state, "deep_cache", None)
    try:
        if cached and cached[0] == key and time.monotonic() - cached[1] < DEEP_TTL:
            registry = cached[2]
        else:
            registry = sha256_of(resolve_bundle_file(*key))
            request.app.state.deep_cache = (key, time.monotonic(), registry)
    except Exception as e:  # noqa: BLE001 - registry fora do ar também é um achado do health check
        response.status_code = 503
        return DeepHealth(status="degraded", model_version=svc.version, loaded_sha256=svc.sha256,
                          registry_sha256=None, detail=f"registry inacessível: {e}")  # fmt: skip
    match = registry == svc.sha256
    response.status_code = 200 if match else 409
    return DeepHealth(
        status="ok" if match else "mismatch", model_version=svc.version, loaded_sha256=svc.sha256,
        registry_sha256=registry,
        detail="ok" if match else "o registry serve outro artefato: reinicie a API para carregar a versão nova",
    )  # fmt: skip
