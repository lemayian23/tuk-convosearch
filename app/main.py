"""
TUK-ConvoSearch application entry point.

Run locally:   uvicorn app.main:app --reload --port 8000
In production: supervisord/uvicorn behind Caddy (see deploy/)

Location: app/main.py
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api import admin, auth_routes, chat, health
from app.config import get_settings
from app.ingestion.index import get_index
from app.services import database
from app.services.auth import ensure_bootstrap_admin
from app.services.ollama import warm_model

logger = logging.getLogger("tuk")

DESCRIPTION = """
Conversational RAG assistant for Technical University of Kenya students,
with a cancer-research literature pipeline alongside it. Questions pass
through Router -> Retriever -> Synthesizer -> Critic, and answers are only
produced from the indexed corpus, with the source document cited.
"""


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    database.init_db()
    ensure_bootstrap_admin()

    # Load the embedder and derive the FAISS index before accepting traffic,
    # so the first student is not charged for a cold start.
    index = await run_in_threadpool(get_index)
    app.state.index = index
    logger.info("Corpus ready: %s chunks", index.get_stats()["total_chunks"])

    if settings.warm_model_on_boot:
        await run_in_threadpool(warm_model, settings.ollama_model, settings.llm_keep_alive)

    logger.info("%s v%s listening (model=%s)", settings.app_name, settings.version, settings.ollama_model)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Compressing markdown/JSON responses is worth real money here: students
    # pay per megabyte, and the admin tables are text-heavy.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again."},
        )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(auth_routes.router)
    app.include_router(admin.router)

    _mount_static(app, settings)
    return app


def _mount_static(app: FastAPI, settings) -> None:
    """
    Serve the built student UI and admin panel from the API origin when the
    bundles exist. Same-origin means no CORS gymnastics for a real deploy;
    when they are absent, / redirects to the API docs so the server is still
    useful.

    Checking for index.html rather than the directory matters on a VPS: Docker
    creates an empty host directory for a bind mount whose source was never
    rsynced, and mounting StaticFiles on that would serve a bare 404 instead of
    the redirect.
    """
    if (settings.admin_dist_dir / "index.html").is_file():
        app.mount(
            "/admin",
            StaticFiles(directory=settings.admin_dist_dir, html=True),
            name="admin",
        )

    if (settings.web_dist_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=settings.web_dist_dir, html=True), name="student")
    else:
        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/docs")


app = create_app()
