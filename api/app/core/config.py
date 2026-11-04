from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração via variáveis de ambiente (ver .env.example na raiz)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Diretório com bundle.joblib, ou URI do MLflow (models:/resonance@production).
    model_registry_uri: str = "../model/production"
    model_version_pin: str = ""  # vazio = alias "production" (latest)
    model_sha256: str = ""  # se definido, recusa um bundle com outro hash (defesa contra artefato adulterado)
    rate_limit_per_minute: int = 120  # por IP, rotas /v1 exceto health; 0 desliga
    data_path: str = "../data/processed/tracks.csv"
    prediction_log_dir: str = "../data/prediction_logs"  # "" desliga o log de predições
    log_level: str = Field("INFO", validation_alias="API_LOG_LEVEL")
    max_batch_size: int = 1000
    cors_origins: list[str] = ["http://localhost:5173"]


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # campos vêm de variáveis de ambiente
