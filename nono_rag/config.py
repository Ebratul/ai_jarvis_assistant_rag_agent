"""Runtime configuration for the NONO RAG agent (read from environment / .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: os.getenv("RAG_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("RAG_PORT", "8100")))
    api_key: str = field(default_factory=lambda: os.getenv("RAG_API_KEY", ""))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("NONO_DATA_DIR", "./data")).resolve())
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "hash").lower())
    fastembed_model: str = field(
        default_factory=lambda: os.getenv(
            "FASTEMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    hf_embed_model: str = field(
        default_factory=lambda: os.getenv(
            "HF_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    backup_keep: int = field(default_factory=lambda: int(os.getenv("BACKUP_KEEP", "30")))
    cors_origins: list[str] = field(default_factory=lambda: _list(os.getenv("CORS_ORIGINS", "*")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "brain.db"

    @property
    def backup_dir(self) -> Path:
        return self.data_dir / "backups"


settings = Settings()
