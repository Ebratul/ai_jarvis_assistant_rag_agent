"""Builds the 3D neural-network view of NONO's memory.

Node types
  core      – NONO itself (the centre of the brain)
  category  – what kind of instruction (reminder, calendar, telegram, ...)
  topic     – recurring keywords shared by ≥2 instructions
  year      – time hubs so you can travel through decades of instructions
  memory    – every single instruction you ever gave
Link types: hub (core→category), category, topic, time, similar (semantic neighbours)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from . import classify
from .store import MemoryStore

TYPE_COLORS = {
    "core": "#7dd3fc",
    "category": "#f8fafc",
    "topic": "#eab308",
    "year": "#a855f7",
}

_cache: dict[tuple, dict[str, Any]] = {}


def _short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_graph(
    store: MemoryStore,
    *,
    limit: int = 1500,
    year_from: int | None = None,
    year_to: int | None = None,
    categories: str | None = None,
    similarity: float = 0.35,
    k: int = 3,
    topics: bool = True,
    years: bool = True,
) -> dict[str, Any]:
    key = (store.version, limit, year_from, year_to, categories, round(similarity, 3), k, topics, years)
    if key in _cache:
        return _cache[key]

    mems = store.list(limit=limit, category=categories, year_from=year_from, year_to=year_to)
    nodes: list[dict[str, Any]] = [
        {"id": "core", "label": "N.O.N.O.", "type": "core", "color": TYPE_COLORS["core"], "val": 18}
    ]
    links: list[dict[str, Any]] = []

    cat_counts = Counter(m["category"] for m in mems)
    for cat, n in cat_counts.items():
        nodes.append({
            "id": f"cat:{cat}", "label": cat.title(), "type": "category", "category": cat,
            "color": classify.category_color(cat), "val": 6 + 2 * math.log1p(n), "count": n,
        })
        links.append({"source": "core", "target": f"cat:{cat}", "type": "hub", "value": 1.0})

    topic_counts = Counter(t for m in mems for t in m["topics"])
    topic_ids = {t for t, n in topic_counts.items() if n >= 2} if topics else set()
    for t in sorted(topic_ids, key=lambda x: -topic_counts[x])[:250]:
        nodes.append({
            "id": f"topic:{t}", "label": t, "type": "topic", "color": TYPE_COLORS["topic"],
            "val": 2.5 + 1.5 * math.log1p(topic_counts[t]), "count": topic_counts[t],
        })
    kept_topics = {n["id"] for n in nodes if n["type"] == "topic"}

    year_counts = Counter(m["created_at"][:4] for m in mems)
    if years:
        for y, n in sorted(year_counts.items()):
            nodes.append({
                "id": f"year:{y}", "label": y, "type": "year", "color": TYPE_COLORS["year"],
                "val": 5 + 1.5 * math.log1p(n), "count": n,
            })
            links.append({"source": "core", "target": f"year:{y}", "type": "hub", "value": 0.5})

    for m in mems:
        nodes.append({
            "id": m["id"], "label": _short(m["text"], 48), "type": "memory", "category": m["category"],
            "color": m["color"], "val": 1.2 + 2.5 * m["importance"] + (2 if m["pinned"] else 0),
            "text": m["text"], "response": _short(m["response"] or "", 600), "created_at": m["created_at"],
            "source": m["source"], "model": m["model"], "topics": m["topics"], "entities": m["entities"],
            "tools": m["tools"], "pinned": m["pinned"],
        })
        links.append({"source": m["id"], "target": f"cat:{m['category']}", "type": "category", "value": 0.6})
        for t in m["topics"]:
            if f"topic:{t}" in kept_topics:
                links.append({"source": m["id"], "target": f"topic:{t}", "type": "topic", "value": 0.4})
        if years:
            links.append({"source": m["id"], "target": f"year:{m['created_at'][:4]}", "type": "time", "value": 0.15})

    # Semantic synapses between instructions.
    ids = [m["id"] for m in mems]
    vecs = store.vectors_for(ids)
    similar = 0
    if len(ids) > 1 and vecs.shape[0] == len(ids):
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1)
        seen: set[tuple[int, int]] = set()
        kk = min(k, len(ids) - 1)
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk] if kk > 0 else np.zeros((len(ids), 0), int)
        for i, row in enumerate(top):
            for j in row:
                s = float(sims[i, j])
                pair = (min(i, int(j)), max(i, int(j)))
                if s < similarity or pair in seen:
                    continue
                seen.add(pair)
                links.append({"source": ids[pair[0]], "target": ids[pair[1]], "type": "similar", "value": round(s, 3)})
                similar += 1

    counts = Counter(n["type"] for n in nodes)
    hubs = sorted(
        (n for n in nodes if n["type"] in ("category", "topic")), key=lambda n: -n.get("count", 0)
    )[:10]
    result = {
        "nodes": nodes,
        "links": links,
        "counts": dict(counts),
        "category_counts": dict(cat_counts),
        "category_colors": {c: classify.category_color(c) for c in classify.CATEGORIES},
        "top_hubs": [{"id": h["id"], "label": h["label"], "count": h.get("count", 0), "color": h["color"]} for h in hubs],
        "total_memories": store.count(),
        "shown_memories": len(mems),
        "similar_links": similar,
        "years": sorted(year_counts),
    }
    if len(_cache) > 32:
        _cache.clear()
    _cache[key] = result
    return result
