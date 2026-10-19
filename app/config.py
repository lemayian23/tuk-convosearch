"""
Central runtime configuration.

Every value that previously lived as a hardcoded literal inside an agent,
service or route is read from the environment here instead, so the same
image can run on a laptop, on a 4GB VPS with llama3.2:1b, or on a 32GB
box with an 8B model — without a code change.

Location: app/config.py
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TUK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_name: str = "TUK-ConvoSearch"
    version: str = "4.0.0"
    environment: Literal["development", "production", "test"] = "development"
    public_url: str = "http://localhost:8000"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    data_dir: Path = REPO_ROOT / "data"
    corpus_dir: Path = REPO_ROOT / "corpus" / "tuk"
    pubmed_index_dir: Path = REPO_ROOT / "pubmed_index"
    max_upload_mb: int = 15
    web_dist_dir: Path = REPO_ROOT / "web" / "dist"
    admin_dist_dir: Path = REPO_ROOT / "admin" / "dist"
    warm_model_on_boot: bool = True

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #
    secret_key: str = ""
    access_token_expire_minutes: int = 480
    bootstrap_admin_email: str = ""
    bootstrap_admin_password: str = ""

    # ------------------------------------------------------------------ #
    # Ollama
    # ------------------------------------------------------------------ #
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_reasoning_model: str = "llama3.2:3b"
    llm_keep_alive: str = "30m"
    llm_timeout_seconds: float = 120.0
    llm_queue_timeout_seconds: float = 45.0
    max_concurrent_llm_calls: int = 2

    synthesis_num_ctx: int = 4096
    synthesis_num_predict: int = 320
    synthesis_temperature: float = 0.2
    synthesis_repeat_penalty: float = 1.1

    # ------------------------------------------------------------------ #
    # Retrieval / pipeline
    # ------------------------------------------------------------------ #
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 160
    top_k: int = 5
    max_sources: int = 3
    history_turns: int = 4
    critic_mode: Literal["heuristic", "llm"] = "heuristic"
    critic_overlap_threshold: float = 0.4

    # ------------------------------------------------------------------ #
    # Limits
    # ------------------------------------------------------------------ #
    rate_limit_per_minute: int = 20
    rate_limit_burst: int = 8
    max_question_chars: int = 1000
    # Caddy/nginx terminate TLS in front of the app and forward the real
    # client address. Without this, every student would share the proxy's IP
    # and one classroom could exhaust the rate limit for everyone.
    trust_forwarded_for: bool = True
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @field_validator(
        "data_dir", "corpus_dir", "pubmed_index_dir", "web_dist_dir", "admin_dist_dir",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, value):
        return Path(value).expanduser()

    @model_validator(mode="after")
    def _check_production_safety(self):
        if self.environment == "production":
            if len(self.secret_key) < 32:
                raise ValueError(
                    "TUK_SECRET_KEY must be set to a random string of at least 32 "
                    "characters in production (see deploy/.env.example)."
                )
            if "*" in self.cors_origins:
                raise ValueError(
                    "CORS allow-origin '*' is not permitted in production. "
                    "List the exact origins that may call the API."
                )
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tuk_convosearch.db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
