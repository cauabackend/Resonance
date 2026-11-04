import re
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routers import model, predict, tracks
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.rate_limit import RateLimiter
from app.models_registry.loader import load_bundle
from app.services.inference import InferenceService
from app.services.prediction_log import PredictionLogger
from app.services.search import CatalogService

log = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.prediction_logger = PredictionLogger(settings.prediction_log_dir)
        app.state.inference = app.state.catalog = None
        # Degradado em vez de crash: /v1/health reporta o que falta, rotas dependentes respondem 503.
        try:
            bundle, digest = load_bundle(settings.model_registry_uri, settings.model_version_pin, settings.model_sha256)
            app.state.inference = InferenceService(bundle, digest)
            log.info("model_loaded", model_version=app.state.inference.version)
        except Exception as e:  # noqa: BLE001
            log.error("model_load_failed", error=str(e))
        try:
            app.state.catalog = CatalogService.from_csv(settings.data_path)
        except Exception as e:  # noqa: BLE001
            log.error("catalog_load_failed", error=str(e))
        yield

    app = FastAPI(title="Resonance API", version="1.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    limiter = RateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        request_id = rid if re.fullmatch(r"[A-Za-z0-9-]{1,64}", rid) else uuid.uuid4().hex
        inference = getattr(request.app.state, "inference", None)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id, model_version=inference.version if inference else None
        )
        request.state.request_id = request_id
        client = request.client.host if request.client else "unknown"
        if request.url.path.startswith("/v1") and request.url.path != "/v1/health" and not limiter.allow(client):
            return JSONResponse(
                {"detail": "Muitas requisições; tente de novo em instantes."},
                status_code=429,
                headers={"Retry-After": "60", "x-request-id": request_id},
            )
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        return response

    for r in (predict.router, tracks.router, model.router):
        app.include_router(r, prefix="/v1")
    return app


app = create_app()
