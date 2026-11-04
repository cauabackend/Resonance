from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_catalog
from app.schemas.track import TrackOut
from app.services.search import CatalogService

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.get("/search", response_model=list[TrackOut])
def search(
    q: str = Query(min_length=2, max_length=100),
    limit: int = Query(10, ge=1, le=50),
    catalog: CatalogService = Depends(get_catalog),
):
    return catalog.search(q, limit)


@router.get("/{track_id}", response_model=TrackOut)
def get_track(track_id: str, catalog: CatalogService = Depends(get_catalog)):
    track = catalog.get(track_id)
    if track is None:
        raise HTTPException(404, "Música não encontrada")
    return track
