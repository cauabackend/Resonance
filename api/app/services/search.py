from itertools import islice
from pathlib import Path

import pandas as pd

from app.schemas.track import TrackFeatures, TrackOut

FEATURES = list(TrackFeatures.model_fields)


def _clean(v):
    """NaN do CSV vira None (campos opcionais); numpy -> tipos nativos."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


class CatalogService:
    """Busca por nome/artista sobre o dataset processado, com índice de texto pré-computado."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.dropna(subset=["track_id"]).drop_duplicates("track_id").reset_index(drop=True)
        self._text = (self.df["name"] + " " + self.df["artist"]).str.lower().to_numpy()
        self._by_id = pd.Series(self.df.index, index=self.df["track_id"])

    @classmethod
    def from_csv(cls, path: str) -> "CatalogService":
        cols = ["track_id", "name", "artist", "popularity", *FEATURES]
        return cls(pd.read_csv(Path(path), usecols=lambda c: c in cols))

    def _out(self, row: pd.Series) -> TrackOut:
        return TrackOut(
            track_id=row["track_id"],
            name=row["name"],
            artist=row["artist"],
            popularity=int(row["popularity"]),
            features=TrackFeatures(**{f: _clean(row.get(f)) for f in FEATURES}),
        )

    def search(self, q: str, limit: int = 10) -> list[TrackOut]:
        needle = q.lower()
        hits = list(islice((i for i, t in enumerate(self._text) if needle in t), limit))
        return [self._out(self.df.iloc[i]) for i in hits]

    def get(self, track_id: str) -> TrackOut | None:
        i = self._by_id.get(track_id)
        return None if i is None else self._out(self.df.iloc[int(i)])
