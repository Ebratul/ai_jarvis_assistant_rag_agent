"""Permanent memory store: SQLite (WAL) + FTS5 trigram index + in-memory vector matrix."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import classify
from .embeddings import Embedder

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    response    TEXT,
    kind        TEXT NOT NULL DEFAULT 'instruction',
    category    TEXT NOT NULL DEFAULT 'general',
    topics      TEXT NOT NULL DEFAULT '[]',
    entities    TEXT NOT NULL DEFAULT '[]',
    tools       TEXT NOT NULL DEFAULT '[]',
    source      TEXT NOT NULL DEFAULT 'text',
    model       TEXT,
    lang        TEXT,
    importance  REAL NOT NULL DEFAULT 0.5,
    pinned      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    embedding   BLOB,
    embed_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(category);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, response, topics, content='memories', content_rowid='rowid', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, text, response, topics) VALUES (new.rowid, new.text, new.response, new.topics);
END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, text, response, topics) VALUES ('delete', old.rowid, old.text, old.response, old.topics);
END;
CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, text, response, topics) VALUES ('delete', old.rowid, old.text, old.response, old.topics);
  INSERT INTO memories_fts(rowid, text, response, topics) VALUES (new.rowid, new.text, new.response, new.topics);
END;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

PUBLIC_FIELDS = (
    "id", "text", "response", "kind", "category", "topics", "entities", "tools", "source",
    "model", "lang", "importance", "pinned", "created_at", "updated_at",
)
_JSON_FIELDS = ("topics", "entities", "tools")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def detect_lang(text: str) -> str:
    bn = sum(1 for ch in text if "ঀ" <= ch <= "৿")
    return "bn" if bn > max(2, len(text) * 0.2) else "en"


class MemoryStore:
    def __init__(self, path: Path, embedder: Embedder):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.embedder = embedder
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self.version = 0  # bumped on every write; used for graph caching
        self._ids: list[str] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self._reembed_mismatched()
        self._load_matrix()

    # ------------------------------------------------------------------ internals
    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        d = {k: row[k] for k in PUBLIC_FIELDS}
        for k in _JSON_FIELDS:
            d[k] = json.loads(d[k] or "[]")
        d["pinned"] = bool(d["pinned"])
        d["color"] = classify.category_color(d["category"])
        return d

    def _load_matrix(self) -> None:
        rows = self._db.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL ORDER BY created_at"
        ).fetchall()
        self._ids = [r["id"] for r in rows]
        if rows:
            self._matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        else:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def _reembed_mismatched(self) -> int:
        """Re-embed rows written by a different embedder (e.g. after switching backend)."""
        rows = self._db.execute(
            "SELECT id, text, response FROM memories WHERE embed_model IS NOT ? OR embedding IS NULL",
            (self.embedder.name,),
        ).fetchall()
        for start in range(0, len(rows), 256):
            chunk = rows[start : start + 256]
            vecs = self.embedder.embed([self._embed_text(r["text"], r["response"]) for r in chunk])
            self._db.executemany(
                "UPDATE memories SET embedding=?, embed_model=? WHERE id=?",
                [(v.tobytes(), self.embedder.name, r["id"]) for v, r in zip(vecs, chunk)],
            )
        self._db.commit()
        return len(rows)

    @staticmethod
    def _embed_text(text: str, response: str | None) -> str:
        # The instruction dominates; a short slice of the answer adds context.
        return text if not response else f"{text}\n{text}\n{response[:300]}"

    # ------------------------------------------------------------------ writes
    def add(
        self,
        text: str,
        *,
        response: str | None = None,
        kind: str = "instruction",
        category: str | None = None,
        source: str = "text",
        model: str | None = None,
        tools: list[str] | None = None,
        created_at: str | None = None,
        topics: list[str] | None = None,
        importance: float | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("text is empty")
        cat = category or classify.classify(text)
        tops = topics or classify.topics(text)
        ents = classify.entities(text)
        imp = importance if importance is not None else classify.importance(text, cat)
        now = utcnow()
        mid = memory_id or uuid.uuid4().hex
        vec = self.embedder.embed([self._embed_text(text, response)])[0]
        with self._lock:
            self._db.execute(
                """INSERT INTO memories (id, text, response, kind, category, topics, entities, tools, source,
                   model, lang, importance, pinned, created_at, updated_at, embedding, embed_model)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?)""",
                (
                    mid, text, response, kind, cat, json.dumps(tops, ensure_ascii=False),
                    json.dumps(ents, ensure_ascii=False), json.dumps(tools or []), source, model,
                    detect_lang(text), imp, created_at or now, now, vec.tobytes(), self.embedder.name,
                ),
            )
            self._db.commit()
            self._ids.append(mid)
            self._matrix = np.vstack([self._matrix, vec[None, :]])
            self.version += 1
        return self.get(mid)  # type: ignore[return-value]

    def update(self, memory_id: str, **fields: Any) -> dict[str, Any] | None:
        current = self.get(memory_id)
        if not current:
            return None
        allowed = {"response", "category", "model", "tools", "pinned", "importance", "kind", "topics"}
        sets, values = [], []
        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            if k in _JSON_FIELDS:
                v = json.dumps(v, ensure_ascii=False)
            if k == "pinned":
                v = int(bool(v))
            sets.append(f"{k}=?")
            values.append(v)
        if not sets:
            return current
        with self._lock:
            self._db.execute(
                f"UPDATE memories SET {', '.join(sets)}, updated_at=? WHERE id=?",
                (*values, utcnow(), memory_id),
            )
            if "response" in fields and fields["response"]:
                vec = self.embedder.embed([self._embed_text(current["text"], fields["response"])])[0]
                self._db.execute("UPDATE memories SET embedding=? WHERE id=?", (vec.tobytes(), memory_id))
                if memory_id in self._ids:
                    self._matrix[self._ids.index(memory_id)] = vec
            self._db.commit()
            self.version += 1
        return self.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._db.commit()
            if cur.rowcount and memory_id in self._ids:
                idx = self._ids.index(memory_id)
                self._ids.pop(idx)
                self._matrix = np.delete(self._matrix, idx, axis=0)
            self.version += 1
            return bool(cur.rowcount)

    def import_many(self, items: Iterable[dict[str, Any]]) -> int:
        n = 0
        for item in items:
            if not item.get("text"):
                continue
            if item.get("id") and self.get(item["id"]):
                continue
            self.add(
                item["text"], response=item.get("response"), kind=item.get("kind", "instruction"),
                category=item.get("category"), source=item.get("source", "import"), model=item.get("model"),
                tools=item.get("tools"), created_at=item.get("created_at"), topics=item.get("topics"),
                importance=item.get("importance"), memory_id=item.get("id"),
            )
            n += 1
        return n

    # ------------------------------------------------------------------ reads
    def get(self, memory_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            f"SELECT {', '.join(PUBLIC_FIELDS)} FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        return self._row(row) if row else None

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        source: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        q: str | None = None,
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        where, args = self._filters(category, source, year_from, year_to)
        if q:
            ids = self._keyword_ids(q, 500)
            if not ids:
                return []
            where.append(f"id IN ({','.join('?' * len(ids))})")
            args += ids
        sql = f"SELECT {', '.join(PUBLIC_FIELDS)} FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY created_at {'ASC' if order == 'asc' else 'DESC'} LIMIT ? OFFSET ?"
        return [self._row(r) for r in self._db.execute(sql, (*args, limit, offset)).fetchall()]

    @staticmethod
    def _filters(category, source, year_from, year_to) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        args: list[Any] = []
        if category:
            cats = [c for c in category.split(",") if c]
            where.append(f"category IN ({','.join('?' * len(cats))})")
            args += cats
        if source:
            where.append("source=?")
            args.append(source)
        if year_from:
            where.append("created_at >= ?")
            args.append(f"{year_from:04d}-01-01")
        if year_to:
            where.append("created_at < ?")
            args.append(f"{year_to + 1:04d}-01-01")
        return where, args

    def _keyword_ids(self, q: str, limit: int) -> list[str]:
        q = q.strip()
        if len(q) >= 3:
            phrase = '"' + q.replace('"', '""') + '"'
            try:
                rows = self._db.execute(
                    "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid "
                    "WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?",
                    (phrase, limit),
                ).fetchall()
                if rows:
                    return [r[0] for r in rows]
            except sqlite3.OperationalError:
                pass
        like = f"%{q}%"
        rows = self._db.execute(
            "SELECT id FROM memories WHERE text LIKE ? OR response LIKE ? ORDER BY created_at DESC LIMIT ?",
            (like, like, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def search(self, query: str, k: int = 6, min_score: float = 0.12, category: str | None = None) -> list[dict[str, Any]]:
        """Hybrid search: cosine similarity + keyword hit bonus + light recency/importance prior."""
        if not query.strip() or not self._ids:
            return []
        qv = self.embedder.embed([query])[0]
        with self._lock:
            sims = self._matrix @ qv
            ids = list(self._ids)
        keyword_hits = set(self._keyword_ids(query, 50))
        scores = {ids[i]: float(sims[i]) for i in np.argsort(-sims)[: max(k * 8, 40)]}
        for mid in keyword_hits:
            scores[mid] = scores.get(mid, 0.0) + 0.25
        results = []
        for mid, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            mem = self.get(mid)
            if not mem or (category and mem["category"] != category):
                continue
            score = s + 0.05 * mem["importance"] + (0.1 if mem["pinned"] else 0)
            if score < min_score:
                continue
            results.append({"memory": mem, "score": round(score, 4)})
            if len(results) >= k:
                break
        return results

    def related(self, memory_id: str, k: int = 6) -> list[dict[str, Any]]:
        if memory_id not in self._ids:
            return []
        with self._lock:
            idx = self._ids.index(memory_id)
            sims = self._matrix @ self._matrix[idx]
            ids = list(self._ids)
        out = []
        for i in np.argsort(-sims):
            if ids[i] == memory_id:
                continue
            mem = self.get(ids[i])
            if mem:
                out.append({"memory": mem, "score": round(float(sims[i]), 4)})
            if len(out) >= k:
                break
        return out

    def vectors_for(self, ids: list[str]) -> np.ndarray:
        with self._lock:
            pos = {mid: i for i, mid in enumerate(self._ids)}
            rows = [pos[i] for i in ids if i in pos]
            return self._matrix[rows] if rows else np.zeros((0, self.embedder.dim), dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        db = self._db
        total = self.count()
        by_cat = {r[0]: r[1] for r in db.execute("SELECT category, COUNT(*) FROM memories GROUP BY category ORDER BY 2 DESC")}
        by_year = {r[0]: r[1] for r in db.execute("SELECT substr(created_at,1,4), COUNT(*) FROM memories GROUP BY 1 ORDER BY 1")}
        by_month = [
            {"month": r[0], "count": r[1]}
            for r in db.execute("SELECT substr(created_at,1,7), COUNT(*) FROM memories GROUP BY 1 ORDER BY 1 DESC LIMIT 24")
        ][::-1]
        by_day = [
            {"day": r[0], "count": r[1]}
            for r in db.execute("SELECT substr(created_at,1,10), COUNT(*) FROM memories GROUP BY 1 ORDER BY 1 DESC LIMIT 30")
        ][::-1]
        by_hour = [0] * 24
        for r in db.execute("SELECT CAST(substr(created_at,12,2) AS INTEGER), COUNT(*) FROM memories GROUP BY 1"):
            if r[0] is not None:
                by_hour[r[0]] = r[1]
        by_source = {r[0]: r[1] for r in db.execute("SELECT source, COUNT(*) FROM memories GROUP BY source")}
        topic_counts: dict[str, int] = {}
        for (tops,) in db.execute("SELECT topics FROM memories ORDER BY created_at DESC LIMIT 5000"):
            for t in json.loads(tops or "[]"):
                topic_counts[t] = topic_counts.get(t, 0) + 1
        top_topics = sorted(topic_counts.items(), key=lambda kv: -kv[1])[:15]
        first_last = db.execute("SELECT MIN(created_at), MAX(created_at) FROM memories").fetchone()
        return {
            "total": total,
            "by_category": by_cat,
            "category_colors": {c: classify.category_color(c) for c in classify.CATEGORIES},
            "by_year": by_year,
            "by_month": by_month,
            "by_day": by_day,
            "by_hour": by_hour,
            "by_source": by_source,
            "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
            "first_at": first_last[0],
            "last_at": first_last[1],
            "embedder": self.embedder.name,
        }

    def iter_all(self) -> Iterable[dict[str, Any]]:
        cur = self._db.execute(f"SELECT {', '.join(PUBLIC_FIELDS)} FROM memories ORDER BY created_at")
        for row in cur:
            yield self._row(row)

    def backup_to(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(target)
        with self._lock:
            self._db.backup(dest)
        dest.close()

    def close(self) -> None:
        self._db.close()
