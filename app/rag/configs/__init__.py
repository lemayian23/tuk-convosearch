"""
Pipeline configurations.

Each module assembles a domain-specific Pipeline from the reusable
agents in app.rag.agents. Requires [rag] extras (vector store backends).
"""

try:
    from app.rag.configs.tuk_pipeline import build_tuk_pipeline
    from app.rag.configs.cancer_pipeline import build_cancer_pipeline
    __all__ = ["build_tuk_pipeline", "build_cancer_pipeline"]
except ImportError:
    # [rag] extras not installed
    build_tuk_pipeline = None
    build_cancer_pipeline = None
    __all__ = []