"""Resolve e carrega o artefato versionado: diretório local ou MLflow Model Registry."""

import hashlib
from pathlib import Path

import joblib

BUNDLE = "bundle.joblib"


def resolve_uri(uri: str, pin: str = "") -> str:
    """Com `pin`, troca o alias 'production' (latest) por uma versão específica do registry."""
    if pin and uri.startswith("models:/") and "@" in uri:
        return f"{uri.split('@')[0]}/{pin}"
    return uri


def resolve_bundle_file(uri: str, pin: str = "") -> Path:
    """Caminho local do bundle (baixa do registry MLflow se o URI for models:/)."""
    uri = resolve_uri(uri, pin)
    if uri.startswith("models:/"):
        import mlflow  # import tardio: o modo local não precisa do cliente MLflow

        path = Path(mlflow.artifacts.download_artifacts(uri))
    else:
        path = Path(uri)
    file = path / BUNDLE
    if not file.exists():
        raise FileNotFoundError(f"{file} não encontrado (rode `dvc repro` em pipeline/)")
    return file


def sha256_of(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def load_bundle(uri: str, pin: str = "", expected_sha256: str = "") -> tuple[dict, str]:
    """Carrega o bundle e devolve (bundle, sha256). Com `expected_sha256`, recusa um artefato diferente do esperado
    ANTES de desserializar (joblib é pickle: um arquivo adulterado executaria código ao carregar)."""
    file = resolve_bundle_file(uri, pin)
    digest = sha256_of(file)
    if expected_sha256 and digest != expected_sha256.lower():
        raise RuntimeError(f"sha256 do bundle ({digest[:12]}…) difere do esperado ({expected_sha256[:12]}…)")
    return joblib.load(file), digest
