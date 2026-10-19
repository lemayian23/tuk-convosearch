"""
Health and readiness. Location: app/api/health.py

Split in two because they answer different questions: a load balancer wants
`/health` to say "this process is up", while an operator and the deploy
script want `/api/health` to say "the corpus is indexed and Ollama can serve
the configured model".
"""

import time
from typing import Any, Dict

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.ingestion.index import get_index
from app.services import database
from app.services.ollama import model_status

router = APIRouter(tags=["health"])

_cache: Dict[str, Any] = {"at": 0.0, "payload": None}
_CACHE_SECONDS = 10.0


@router.get("/health")
async def liveness() -> Dict[str, str]:
    return {"status": "alive", "version": get_settings().version}


@router.get("/api/health")
async def readiness() -> Dict[str, Any]:
    now = time.monotonic()
    if now - _cache["at"] < _CACHE_SECONDS and _cache["payload"] is not None:
        return _cache["payload"]

    settings = get_settings()
    index_stats = await run_in_threadpool(get_index().get_stats)
    ollama = await run_in_threadpool(model_status)
    documents = len(database.list_documents())
    stored_chunks = database.chunk_count()

    checks = {
        "corpus_indexed": index_stats["total_chunks"] > 0,
        # Out-of-band ingestion (scripts/ingest_corpus.py) writes rows this
        # process cannot see until it restarts, and students would silently
        # get "nothing matched" answers from a corpus that is actually there.
        "index_matches_corpus": stored_chunks == index_stats["total_chunks"],
        "ollama_reachable": ollama["reachable"],
        "model_present": ollama["model_present"],
    }
    payload = {
        "status": "ready" if all(checks.values()) else "degraded",
        "version": settings.version,
        "environment": settings.environment,
        "public_url": settings.public_url,
        "checks": checks,
        "chunks": index_stats["total_chunks"],
        "stored_chunks": stored_chunks,
        "documents": documents,
        "embedding_model": index_stats["embedding_model"],
        "ollama": {
            "host": settings.ollama_host,
            "expected_model": ollama.get("expected_model"),
            "models": ollama.get("models", []),
            "error": ollama.get("error"),
        },
        "checked_at": time.time(),
    }
    _cache.update(at=now, payload=payload)
    return payload


@router.get("/api/info")
async def info() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "app": settings.app_name,
        "version": settings.version,
        "model": settings.ollama_model,
        "pipeline": "Router -> Retriever -> Synthesizer -> Critic",
        "docs": "/docs",
    }
