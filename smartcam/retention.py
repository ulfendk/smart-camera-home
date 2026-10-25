"""What is kept on disk.

Recordings are a rolling buffer: every segment is kept for `continuous_hours`, long enough for event
clips to be cut from them. What lasts are the events: their clips and snapshots are kept for
`event_days`. If free space still drops below `min_free_gb`, the oldest recordings are deleted first
(never the last hour), then the oldest events.
"""

import asyncio
import logging
import shutil
import time

from .config import Config
from .db import Database
from .recorder import hour_dirs

log = logging.getLogger(__name__)


def free_gb(cfg: Config) -> float:
    return shutil.disk_usage(cfg.paths.data).free / 1e9


def _remove_hour(cfg: Config, hour_dir) -> None:
    shutil.rmtree(hour_dir, ignore_errors=True)
    day_dir = hour_dir.parent
    if day_dir.is_dir() and not any(day_dir.iterdir()):
        day_dir.rmdir()


def _remove_event(cfg: Config, db: Database, event: dict) -> None:
    for key in ("clip", "snapshot", "crop"):
        if event.get(key):
            (cfg.paths.data / event[key]).unlink(missing_ok=True)
    for extra in (cfg.paths.data / "snapshots").glob(f"{event['id']}_*.jpg"):   # speed events' best shots
        extra.unlink(missing_ok=True)
    (cfg.paths.data / "snapshots" / f"{event['id']}.jpg").unlink(missing_ok=True)
    db.delete_event(event["id"])


def prune(cfg: Config, db: Database, now: float | None = None) -> tuple[int, int]:
    """Apply the retention rules. Returns (hours of recordings, events) removed."""
    now = now or time.time()
    r = cfg.record
    removed_hours = removed_events = 0
    for start, hour_dir in hour_dirs(cfg):
        if start + 3600 < now - r.continuous_hours * 3600:
            _remove_hour(cfg, hour_dir)
            removed_hours += 1
    for event in db.events_before(now - r.event_days * 86400):
        _remove_event(cfg, db, event)
        removed_events += 1

    # Disk guard: recordings older than the last hour first, then the oldest events.
    spare = [(s, d) for s, d in hour_dirs(cfg) if s + 3600 < now - 3600]
    while free_gb(cfg) < r.min_free_gb and spare:
        _remove_hour(cfg, spare.pop(0)[1])
        removed_hours += 1
    while free_gb(cfg) < r.min_free_gb and (oldest := db.events_before(limit=1)):
        _remove_event(cfg, db, oldest[0])
        removed_events += 1
    return removed_hours, removed_events


async def run(cfg: Config, db: Database) -> None:
    while True:
        hours, events = await asyncio.to_thread(prune, cfg, db)
        if hours or events:
            log.info("retention: removed %d hour(s) of recordings and %d event(s); %.0f GB free",
                     hours, events, free_gb(cfg))
        await asyncio.sleep(300)
