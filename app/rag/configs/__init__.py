"""
Pipeline configurations.

Each module in this package assembles a domain-specific Pipeline
from the reusable agents in app.rag.agents.
"""

from app.rag.configs.tuk_pipeline import build_tuk_pipeline

__all__ = [
    "build_tuk_pipeline",
]