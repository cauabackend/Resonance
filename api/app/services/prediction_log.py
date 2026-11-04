import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class PredictionLogger:
    """JSONL particionado por data (dt=YYYY-MM-DD): entrada dos relatórios de drift (features e percentile_score)."""

    def __init__(self, root: str):
        self.root = Path(root) if root else None
        self._lock = threading.Lock()

    def log(self, request_id: str, model_version: str, rows: list[tuple[dict, float, float, bool, str]]) -> None:
        """rows: (features, percentile_score, probability, is_hit, mode)."""
        if self.root is None:
            return
        now = datetime.now(UTC)
        lines = "".join(
            json.dumps(
                {
                    "ts": now.isoformat(),
                    "request_id": request_id,
                    "model_version": model_version,
                    "features": feats,
                    "percentile_score": score,
                    "probability": prob,
                    "is_hit": hit,
                    "mode": mode,
                },
                allow_nan=False,
            )
            + "\n"
            for feats, score, prob, hit, mode in rows
        )
        part = self.root / f"dt={now:%Y-%m-%d}"
        with self._lock:
            part.mkdir(parents=True, exist_ok=True)
            with (part / "predictions.jsonl").open("a", encoding="utf-8") as f:
                f.write(lines)
