"""FastAPI service: POST /search, POST /ask and GET /health."""
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.answer import answer_query
from app.config import Settings, get_settings
from app.db import connect
from app.embeddings import Embedder
from app.rerank import Reranker
from app.retrieval import search


INDEX_HTML = Path(__file__).parent / "static" / "index.html"
NO_RESULTS_ANSWER = "No similar past RFQs were found."


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SearchRequest(AskRequest):
    mode: Literal["dense", "lexical", "hybrid"] = "hybrid"
    rerank: bool = False


def create_app(
    conn_factory: Callable[[], Any],
    embedder: Embedder,
    claude_client: Any | None,
    settings: Settings,
    reranker_factory: Callable[[], Reranker] | None = None,
) -> FastAPI:
    app = FastAPI(title="RFQ similarity search")
    app.state.claude_client = claude_client
    app.state.settings = settings
    app.state.reranker = None
    reranker_lock = threading.Lock()

    def get_reranker() -> Reranker:
        if app.state.reranker is None:
            with reranker_lock:
                if app.state.reranker is None:
                    app.state.reranker = reranker_factory()
        return app.state.reranker

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.post("/search")
    def search_endpoint(req: SearchRequest):
        extra: dict[str, Any] = {}
        if req.rerank:
            if reranker_factory is None:
                raise HTTPException(503, "reranker not configured")
            extra = {"reranker": get_reranker(), "rerank_depth": settings.rerank_depth}
        conn = conn_factory()
        try:
            hits = search(
                conn, embedder, req.query, k=req.k, mode=req.mode,
                lexical_weight=settings.hybrid_lexical_weight, rrf_k=settings.rrf_k,
                **extra,
            )
        finally:
            conn.close()
        return {"results": [h.to_dict() for h in hits]}

    @app.post("/ask")
    def ask_endpoint(req: AskRequest):
        if app.state.claude_client is None:
            raise HTTPException(503, "ANTHROPIC_API_KEY not set; /search still works")
        conn = conn_factory()
        try:
            hits = search(
                conn, embedder, req.query, k=req.k, mode="hybrid",
                lexical_weight=settings.hybrid_lexical_weight, rrf_k=settings.rrf_k,
            )
        finally:
            conn.close()
        if not hits:
            return {"answer": NO_RESULTS_ANSWER, "citations": [], "sources": []}
        try:
            result = answer_query(
                app.state.claude_client, settings.claude_model, req.query, hits
            )
        except anthropic.APIError:
            raise HTTPException(502, "upstream model error")
        return {
            "answer": result.answer,
            "citations": [asdict(c) for c in result.citations],
            "sources": [h.to_dict() for h in hits],
        }

    @app.get("/health")
    def health():
        conn = conn_factory()
        try:
            n_rfqs = conn.execute("SELECT count(*) FROM rfqs").fetchone()[0]
            n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()
        return {
            "status": "ok",
            "rfqs": n_rfqs,
            "chunks": n_chunks,
            "embedding_model": settings.embedding_model,
        }

    return app


def _build_default_app() -> FastAPI:
    from app.embeddings import E5Embedder

    settings = get_settings()
    client = None
    if settings.anthropic_api_key:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    factory = None
    if settings.reranker_model:
        from app.rerank import CrossEncoderReranker

        def factory() -> Reranker:
            return CrossEncoderReranker(settings.reranker_model)

    return create_app(
        lambda: connect(settings.database_url),
        E5Embedder(settings.embedding_model),
        client,
        settings,
        reranker_factory=factory,
    )


_app: FastAPI | None = None


def __getattr__(name: str):
    # Lazy module-level `app` so importing app.main does not load E5.
    global _app
    if name == "app":
        if _app is None:
            _app = _build_default_app()
        return _app
    raise AttributeError(name)
