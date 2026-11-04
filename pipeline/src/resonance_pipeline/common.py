from pathlib import Path

import yaml
from resonance_features import AUDIO

ROOT = Path(__file__).resolve().parents[3]

FEATURES = AUDIO  # as 9 features de áudio originais (a lista vive em resonance_features)
GENRE = "track_genre"
TARGET = "is_hit"


def load_params(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "pipeline" / "params.yaml").read_text(encoding="utf-8"))


def repo_path(p: str | Path) -> Path:
    """Caminhos do params.yaml são relativos à raiz do repo, não ao cwd do DVC."""
    return Path(p) if Path(p).is_absolute() else ROOT / p
