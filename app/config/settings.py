"""Typed environment-backed configuration for BrainSeg."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_prefix="BRAINSEG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(
        default="development",
        validation_alias=AliasChoices("BRAINSEG_ENV", "ENVIRONMENT"),
    )
    log_level: str = "INFO"
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    random_seed: int = 42

    kaggle_username: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KAGGLE_USERNAME", "BRAINSEG_KAGGLE_USERNAME"),
    )
    kaggle_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KAGGLE_KEY", "BRAINSEG_KAGGLE_KEY"),
    )
    kaggle_dataset: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KAGGLE_DATASET", "BRAINSEG_KAGGLE_DATASET"),
    )

    database_url: str = "postgresql+psycopg://brainseg:brainseg@localhost:5432/brainseg"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: Path = Path("artifacts/mlflow")

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def require_kaggle_credentials(self) -> tuple[str, str, str]:
        """Return Kaggle settings or raise a useful error at download time."""

        missing = [
            name
            for name, value in {
                "KAGGLE_USERNAME": self.kaggle_username,
                "KAGGLE_KEY": self.kaggle_key,
                "KAGGLE_DATASET": self.kaggle_dataset,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing Kaggle settings: "
                + ", ".join(missing)
                + ". Set them in an ignored .env file before downloading data."
            )
        return self.kaggle_username, self.kaggle_key, self.kaggle_dataset


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""

    return Settings()
