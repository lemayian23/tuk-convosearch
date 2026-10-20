"""
Pipeline construction from runtime settings.

Pipelines are expensive: the TU-K one holds the embedding model and the FAISS
index, both shared process-wide. They are built lazily, once per pipeline
name, and reused for every request.

Location: app/services/pipelines.py
"""

import logging
import threading
from pathlib import Path
from typing import Dict

from app.config import Settings, get_settings
from app.rag.configs.cancer_pipeline import build_cancer_pipeline
from app.rag.configs.tuk_pipeline import build_tuk_pipeline
from app.rag.core.pipeline import Pipeline

logger = logging.getLogger(__name__)

_cache: Dict[str, Pipeline] = {}
_lock = threading.Lock()


class PipelineUnavailableError(RuntimeError):
    """The requested pipeline cannot run on this server."""


def _pubmed_corpus_present(settings: Settings) -> bool:
    """
    True only when the local PubMed snapshot is actually installed.

    PubMedStore does not fail on a missing index - it builds an empty one, so
    an unmounted pubmed_index/ gives a student twenty seconds of thinking and
    then "nothing found", plus a second copy of the embedding model in RAM.
    Better to say the corpus is not here.
    """
    corpus = Path(settings.pubmed_index_dir)
    return (corpus / "faiss_index.bin").is_file() and (corpus / "metadata.json").is_file()


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
            max_chunks_per_source=settings.max_chunks_per_source,
            min_relevance_margin=settings.min_relevance_margin,
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

    if key == "cancer" and not _pubmed_corpus_present(settings):
        raise PipelineUnavailableError(
            "The cancer research corpus is not installed on this server. "
            "Sync pubmed_index/ and uncomment its mount in "
            "deploy/docker-compose.yml to enable it."
        )

    with _lock:
        if key not in _cache:
            logger.info("Building pipeline %r", key)
            _cache[key] = _build(key, settings)
        return _cache[key]


def available_pipelines() -> Dict[str, str]:
    settings = get_settings()
    pipelines = {
        "tuk": "TU-K student assistant (academic calendar, fees, registration, exams)",
    }
    if _pubmed_corpus_present(settings):
        pipelines["cancer"] = "Cancer research literature assistant (PubMed abstracts)"
    return pipelines


def reset_pipelines() -> None:
    global _cache
    with _lock:
        _cache = {}
