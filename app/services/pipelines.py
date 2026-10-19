"""
Pipeline construction from runtime settings.

Pipelines are expensive: the TU-K one holds the embedding model and the FAISS
index, both shared process-wide. They are built lazily, once per pipeline
name, and reused for every request.

Location: app/services/pipelines.py
"""

import logging
import threading
from typing import Dict, Optional

from app.config import Settings, get_settings
from app.rag.configs.cancer_pipeline import build_cancer_pipeline
from app.rag.configs.tuk_pipeline import build_tuk_pipeline
from app.rag.core.pipeline import Pipeline

logger = logging.getLogger(__name__)

_cache: Dict[str, Pipeline] = {}
_lock = threading.Lock()


def _build(name: str, settings: Settings) -> Pipeline:
    if name == "tuk":
        from app.ingestion.index import get_index

        return build_tuk_pipeline(
            store=get_index(),
            model_name=settings.ollama_model,
            critic_mode=settings.critic_mode,
            critic_model=settings.ollama_reasoning_model,
            critic_overlap_threshold=settings.critic_overlap_threshold,
            max_sources=settings.max_sources,
            num_predict=settings.synthesis_num_predict,
            num_ctx=settings.synthesis_num_ctx,
            temperature=settings.synthesis_temperature,
            repeat_penalty=settings.synthesis_repeat_penalty,
            keep_alive=settings.llm_keep_alive,
        )

    if name == "cancer":
        from app.rag.stores.pubmed_store import PubMedStore

        return build_cancer_pipeline(
            store=PubMedStore(persist_directory=str(settings.pubmed_index_dir)),
            synthesizer_model=settings.ollama_model,
            critic_model=settings.ollama_reasoning_model,
            expander_model=settings.ollama_reasoning_model,
        )

    raise KeyError(f"Unknown pipeline: {name!r}")


def get_pipeline(name: str = "tuk") -> Pipeline:
    settings = get_settings()
    key = name if name in ("tuk", "cancer") else "tuk"
    with _lock:
        if key not in _cache:
            logger.info("Building pipeline %r", key)
            _cache[key] = _build(key, settings)
        return _cache[key]


def available_pipelines() -> Dict[str, str]:
    return {
        "tuk": "TU-K student assistant (academic calendar, fees, registration, exams)",
        "cancer": "Cancer research literature assistant (PubMed abstracts)",
    }


def reset_pipelines() -> None:
    global _cache
    with _lock:
        _cache = {}
