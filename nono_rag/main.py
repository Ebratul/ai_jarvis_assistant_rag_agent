"""NONO RAG agent — FastAPI service holding NONO's permanent memory and neural graph."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .backup import run_backup
from .config import settings
from .embeddings import build_embedder
from .graph import build_graph
from .store import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("nono.rag")

store: MemoryStore | None = None


def get_store() -> MemoryStore:
    if store is None:
        raise HTTPException(503, "memory store not ready")
    return store


async def _backup_loop() -> None:
    while True:
        try:
            run_backup(get_store(), settings.backup_dir, settings.backup_keep)
        except Exception as exc:  # pragma: no cover
            log.warning("backup failed: %s", exc)
        await asyncio.sleep(24 * 3600)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global store
    store = MemoryStore(settings.db_path, build_embedder())
    log.info("Brain loaded: %d memories (%s) at %s", store.count(), store.embedder.name, settings.db_path)
    task = asyncio.create_task(_backup_loop())
    yield
    task.cancel()
    store.close()
    store = None


app = FastAPI(title="NONO RAG Agent", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
)


def auth(x_nono_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_nono_key != settings.api_key:
        raise HTTPException(401, "invalid X-Nono-Key")


# ---------------------------------------------------------------- schemas
class MemoryIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    response: str | None = None
    kind: str = "instruction"
    category: str | None = None
    source: str = "text"
    model: str | None = None
    tools: list[str] | None = None
    created_at: str | None = None


class MemoryPatch(BaseModel):
    response: str | None = None
    category: str | None = None
    model: str | None = None
    tools: list[str] | None = None
    pinned: bool | None = None
    importance: float | None = None


class SearchIn(BaseModel):
    query: str
    k: int = Field(default=6, ge=1, le=50)
    min_score: float = 0.12
    category: str | None = None


# ---------------------------------------------------------------- routes
@app.get("/health")
async def health() -> dict[str, Any]:
    s = get_store()
    return {"ok": True, "service": "nono-rag", "version": __version__, "memories": s.count(), "embedder": s.embedder.name}


@app.post("/memories", dependencies=[Depends(auth)])
async def add_memory(body: MemoryIn) -> dict[str, Any]:
    try:
        return get_store().add(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/memories", dependencies=[Depends(auth)])
async def list_memories(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    category: str | None = None,
    source: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    q: str | None = None,
    order: str = "desc",
) -> dict[str, Any]:
    s = get_store()
    items = s.list(limit=limit, offset=offset, category=category, source=source,
                   year_from=year_from, year_to=year_to, q=q, order=order)
    return {"items": items, "total": s.count()}


@app.get("/memories/{memory_id}", dependencies=[Depends(auth)])
async def get_memory(memory_id: str) -> dict[str, Any]:
    s = get_store()
    mem = s.get(memory_id)
    if not mem:
        raise HTTPException(404, "memory not found")
    return {**mem, "related": s.related(memory_id)}


@app.patch("/memories/{memory_id}", dependencies=[Depends(auth)])
async def patch_memory(memory_id: str, body: MemoryPatch) -> dict[str, Any]:
    mem = get_store().update(memory_id, **body.model_dump(exclude_none=True))
    if not mem:
        raise HTTPException(404, "memory not found")
    return mem


@app.delete("/memories/{memory_id}", dependencies=[Depends(auth)])
async def delete_memory(memory_id: str) -> dict[str, bool]:
    if not get_store().delete(memory_id):
        raise HTTPException(404, "memory not found")
    return {"deleted": True}


@app.post("/search", dependencies=[Depends(auth)])
async def search(body: SearchIn) -> dict[str, Any]:
    return {"results": get_store().search(body.query, body.k, body.min_score, body.category)}


@app.get("/graph", dependencies=[Depends(auth)])
async def graph(
    limit: int = Query(1500, ge=1, le=20000),
    year_from: int | None = None,
    year_to: int | None = None,
    categories: str | None = None,
    similarity: float = Query(0.35, ge=0, le=1),
    k: int = Query(3, ge=0, le=12),
    topics: bool = True,
    years: bool = True,
) -> dict[str, Any]:
    return build_graph(get_store(), limit=limit, year_from=year_from, year_to=year_to,
                       categories=categories, similarity=similarity, k=k, topics=topics, years=years)


@app.get("/stats", dependencies=[Depends(auth)])
async def stats() -> dict[str, Any]:
    return get_store().stats()


@app.get("/export", dependencies=[Depends(auth)])
async def export() -> StreamingResponse:
    """Full JSONL backup of every instruction — human-readable and future-proof."""
    def gen():
        for mem in get_store().iter_all():
            mem.pop("color", None)
            yield json.dumps(mem, ensure_ascii=False) + "\n"
    return StreamingResponse(
        gen(), media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="nono-brain.jsonl"'},
    )


@app.post("/import", dependencies=[Depends(auth)])
async def import_memories(request: Request) -> dict[str, int]:
    raw = (await request.body()).decode("utf-8").strip()
    if raw.startswith("["):
        items = json.loads(raw)
    else:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return {"imported": get_store().import_many(items)}


@app.post("/admin/backup", dependencies=[Depends(auth)])
async def backup_now() -> dict[str, str]:
    path = run_backup(get_store(), settings.backup_dir, settings.backup_keep)
    return {"backup": str(path)}


def run() -> None:
    import uvicorn

    uvicorn.run("nono_rag.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
