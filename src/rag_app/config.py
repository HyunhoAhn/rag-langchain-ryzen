"""Configuration loading for the local RAG CLI."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_LEMONADE_BASE_URL = "http://localhost:13305/v1"
DEFAULT_LEMONADE_CHAT_MODEL = "user.Qwen3-8B-GGUF"
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION_NAME = "ryzen_ai_max_rag"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


@dataclass(frozen=True)
class AppConfig:
    lemonade_base_url: str
    lemonade_chat_model: str
    chroma_dir: str
    collection_name: str
    embedding_model: str


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(env_file: str | Path = ".env") -> AppConfig:
    dotenv_values = _read_dotenv(Path(env_file))

    def get(name: str, default: str) -> str:
        return os.environ.get(name) or dotenv_values.get(name) or default

    return AppConfig(
        lemonade_base_url=get("LEMONADE_BASE_URL", DEFAULT_LEMONADE_BASE_URL),
        lemonade_chat_model=get("LEMONADE_CHAT_MODEL", DEFAULT_LEMONADE_CHAT_MODEL),
        chroma_dir=get("CHROMA_DIR", DEFAULT_CHROMA_DIR),
        collection_name=get("COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
        embedding_model=get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
