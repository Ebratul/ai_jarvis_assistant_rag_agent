# NONO · RAG Agent (Neural Memory)

The long-term brain of **NONO**, a personal voice AI assistant. Every instruction you give NONO, by voice, text or Telegram, is stored here **permanently**. Each one is classified, embedded and linked into a **3D neural network** that you can still browse in 10 or 20 years.

```
frontend (3D Brain view) ──► backend ──► RAG agent :8100 ──► data/brain.db (SQLite WAL)
                                                         └─► data/backups/brain-YYYYMMDD.db (daily)
```

## Features
| | |
|---|---|
| **Permanent store** | SQLite (WAL) with FTS5 trigram index (substring search in English, Bangla and Banglish) |
| **Embeddings** | `hash` (default: multilingual, no download, stable for decades) · `fastembed` (multilingual MiniLM) · `huggingface` (Inference API) |
| **Auto-enrichment** | category (reminder, calendar, telegram, system, memory, work, personal, health, finance, learning, web, question, general), topics, entities, importance, language |
| **Hybrid search** | cosine similarity + keyword bonus + importance/pin prior, used by the backend for RAG recall |
| **Neural graph** | `core → categories → memories`, `memories ↔ topics`, `memories → year hubs`, plus semantic synapses between similar instructions |
| **Longevity** | daily snapshot backups (keeps 30), full JSONL export/import, automatic re-embedding when you change embedder |

## Run
```bash
uv venv .venv && uv pip install -p .venv/bin/python -r requirements.txt   # or: pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m nono_rag            # → http://localhost:8100/docs
```
Optional neural embeddings: `pip install -r requirements-ml.txt` and set `EMBEDDING_BACKEND=fastembed`. Existing memories are re-embedded automatically on the next start.

Demo data (use a separate data dir so it never mixes with your real brain):
```bash
NONO_DATA_DIR=./demo-data RAG_PORT=8199 .venv/bin/python -m nono_rag &
.venv/bin/python scripts/seed_demo.py --url http://localhost:8199 --count 220
```

Docker: `docker build -t nono-rag . && docker run -p 8100:8100 -v nono-brain:/data nono-rag`

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, memory count, embedder |
| POST | `/memories` | store an instruction `{text, response?, source?, model?, category?, created_at?}` |
| GET | `/memories` | list with `limit, offset, q, category, source, year_from, year_to, order` |
| GET / PATCH / DELETE | `/memories/{id}` | detail (+ related memories), update response/pin/category, delete |
| POST | `/search` | `{query, k, min_score}` hybrid semantic search |
| GET | `/graph` | 3D graph `{nodes, links, counts, top_hubs, years}`; filters `year_from, year_to, categories, similarity, k, limit` |
| GET | `/stats` | totals by category, year, month, day, hour, source, top topics |
| GET | `/export` · POST `/import` | JSONL backup / restore |
| POST | `/admin/backup` | snapshot now |

If `RAG_API_KEY` is set, every call except `/health` needs the `X-Nono-Key` header.

## Tests
```bash
uv pip install -p .venv/bin/python pytest && .venv/bin/python -m pytest -q
```

## Keeping your memories for 20 years
* Back up the `data/` folder (or the Docker volume): `brain.db` plus `backups/`.
* Run `curl localhost:8100/export > nono-brain.jsonl` now and then. It's plain text, so any future tool can read it.
* The default `hash` embedder is pure math with no model download, so it will produce the same vectors in 2046 as it does today.
