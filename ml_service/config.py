from functools import lru_cache
import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "support_kb"
    embedding_provider: str = "hash"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    data_dir: Path = ROOT_DIR / "data"
    knowledge_dir: Path = ROOT_DIR / "knowledge"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def deepseek_api_key(self) -> str | None:
        values = dotenv_values(ROOT_DIR / ".env")
        return (
            os.environ.get("DEEPSEEK-API-KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or values.get("DEEPSEEK-API-KEY")
            or values.get("DEEPSEEK_API_KEY")
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
