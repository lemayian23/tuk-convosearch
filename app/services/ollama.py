"""
Ollama status helpers. Location: app/services/ollama.py

Plain HTTP against the Ollama API rather than the `ollama` Python client,
because these calls are diagnostics: they must work even when the client or
the model is in a bad state, and they must never be the reason a health
check hangs.
"""

import threading
from typing import Any, Dict, List

import requests

from app.config import get_settings

_THREAD_LOCAL = threading.local()


def client():
    """
    A client bound to TUK_OLLAMA_HOST, one per thread.

    The `ollama` package's module-level `ollama.chat()` reads its host from
    `OLLAMA_HOST`/`http://localhost:11434`, never from our settings, so agents
    that used it would look for the model on localhost and fail on any deploy
    where Ollama is a sibling container. One client per thread also keeps the
    request Sessions off the pipeline's shared thread pool.
    """
    local = getattr(_THREAD_LOCAL, "client", None)
    if local is None:
        import ollama

        local = ollama.Client(host=get_settings().ollama_host.rstrip("/"))
        _THREAD_LOCAL.client = local
    return local


def tags(timeout: float = 4.0) -> Dict[str, Any]:
    host = get_settings().ollama_host.rstrip("/")
    result: Dict[str, Any] = {"reachable": False, "models": [], "error": None}
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
        names: List[str] = [
            model.get("name", "") for model in response.json().get("models", [])
        ]
        result.update(reachable=True, models=names)
    except requests.RequestException as exc:
        result["error"] = str(exc)
    return result


def model_status(timeout: float = 4.0) -> Dict[str, Any]:
    """Is the configured synthesis model installed on the Ollama server?"""
    settings = get_settings()
    status = tags(timeout=timeout)
    wanted = settings.ollama_model
    status["expected_model"] = wanted
    status["model_present"] = any(
        name == wanted or name.split(":")[0] == wanted.split(":")[0]
        for name in status["models"]
    )
    return status


def warm_model(name: str, keep_alive: str = "30m", timeout: float = 90.0) -> bool:
    """
    Pull the model into memory at boot so the first student does not wait
    for a cold load.
    """
    host = get_settings().ollama_host.rstrip("/")
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": name, "prompt": "", "keep_alive": keep_alive},
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False
