from fastapi import HTTPException, Request

from app.services.inference import InferenceService
from app.services.search import CatalogService


def get_inference(request: Request) -> InferenceService:
    svc = request.app.state.inference
    if svc is None:
        raise HTTPException(503, "Modelo não carregado")
    return svc


def get_catalog(request: Request) -> CatalogService:
    svc = request.app.state.catalog
    if svc is None:
        raise HTTPException(503, "Catálogo de músicas não carregado")
    return svc
