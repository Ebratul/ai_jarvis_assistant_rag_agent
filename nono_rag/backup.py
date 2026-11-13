"""Daily snapshot backups so the brain survives for decades."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .store import MemoryStore

log = logging.getLogger("nono.rag.backup")


def run_backup(store: MemoryStore, backup_dir: Path, keep: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = backup_dir / f"brain-{stamp}.db"
    store.backup_to(target)
    snapshots = sorted(backup_dir.glob("brain-*.db"))
    for old in snapshots[: max(0, len(snapshots) - keep)]:
        old.unlink(missing_ok=True)
    log.info("Brain backup written to %s", target)
    return target
