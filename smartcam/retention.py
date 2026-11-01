"""What is kept on disk.

Recordings are a rolling buffer: every segment is kept for `continuous_hours`, long enough for event
clips to be cut from them. What lasts are the events: their clips and snapshots are kept for
`event_days`. If free space still drops below `min_free_gb`, or what SmartCam stores grows past `max_gb`
(for a disk shared with other software), the oldest recordings are deleted first (never the last hour),
then the oldest events.
"""

import asyncio
import logging
import shutil
import sqlite3
import time

from .config import Config
from .db import Database
from .recorder import hour_dirs

log = logging.getLogger(__name__)


MEDIA = ("recordings", "clips", "snapshots", "faces")   # what max_gb counts (the database is small)


def free_gb(cfg: Config) -> float:
    return shutil.disk_usage(cfg.paths.data).free / 1e9


def size_gb(path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e9
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9 if path.is_dir() else 0


def used_gb(cfg: Config) -> float:
    return sum(size_gb(cfg.paths.data / d) for d in MEDIA)


def _event_gb(cfg: Config, event: dict) -> float:
    return sum(size_gb(cfg.paths.data / event[k]) for k in ("clip", "snapshot", "crop") if event.get(k))


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
    _remove_faces(cfg, db, event["id"])
    db.delete_event(event["id"])


def _remove_faces(cfg: Config, db: Database, event_id: int) -> None:
    """The event's unnamed faces go with it; named ones are what SmartCam has learned, and stay."""
    try:
        with db.lock, db.conn:
            crops = db.conn.execute("select crop from faces where event_id = ? and person is null", (event_id,)).fetchall()
            db.conn.execute("delete from faces where event_id = ? and person is null", (event_id,))
    except sqlite3.OperationalError:
        return   # no faces table: recognition never ran
    for (crop,) in crops:
        if crop:
            (cfg.paths.data / crop).unlink(missing_ok=True)


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

    # Disk guard: recordings older than the last hour first, then the oldest events. What is stored is
    # measured once and counted down, rather than walking every file again after each removal.
    used = used_gb(cfg) if r.max_gb is not None else 0
    short = lambda: free_gb(cfg) < r.min_free_gb or (r.max_gb is not None and used > r.max_gb)
    spare = [(s, d) for s, d in hour_dirs(cfg) if s + 3600 < now - 3600]
    while short() and spare:
        hour_dir = spare.pop(0)[1]
        used -= size_gb(hour_dir)
        _remove_hour(cfg, hour_dir)
        removed_hours += 1
    while short() and (oldest := db.events_before(limit=1)):
        used -= _event_gb(cfg, oldest[0])
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
