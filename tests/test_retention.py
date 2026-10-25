import tempfile
from datetime import datetime
from pathlib import Path

from smartcam import retention
from smartcam.config import load_config
from smartcam.db import Database

TEST_CONFIG = Path(__file__).parent / "config.yml"
NOW = datetime(2026, 10, 25, 12, 30).timestamp()
HOUR = 3600


def setup(monkeypatch, free_gb: float = 1000):
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    for age in range(0, 10):               # one segment per hour, 0..9 hours old
        t = datetime.fromtimestamp(NOW - age * HOUR)
        seg = cfg.paths.recordings / t.strftime("%Y-%m-%d/%H") / "front_door" / "00.00.mp4"
        seg.parent.mkdir(parents=True, exist_ok=True)
        seg.write_bytes(b"x")
    events = []
    for age_days in (1, 5, 12):
        eid = db.insert_event(camera="front_door", label="person", start_ts=NOW - age_days * 86400, max_score=0.9)
        clip = cfg.paths.data / "clips" / f"{eid}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"x")
        db.update_event(eid, clip=f"clips/{eid}.mp4")
        events.append(eid)
    monkeypatch.setattr(retention, "free_gb", lambda cfg: free_gb)
    return cfg, db, events


def hours_left(cfg) -> int:
    return len(list(cfg.paths.recordings.glob("*/*")))


def test_keeps_continuous_buffer_and_recent_events(monkeypatch):
    cfg, db, events = setup(monkeypatch)
    cfg.record.continuous_hours, cfg.record.event_days = 6, 10
    hours, removed = retention.prune(cfg, db, NOW)
    # Hour dirs that ended more than 6 h ago go; the 12-day-old event goes with its clip.
    assert hours_left(cfg) == 7 and removed == 1
    assert [e["id"] for e in db.events_before()] == events[1::-1]
    assert not (cfg.paths.data / "clips" / f"{events[2]}.mp4").exists()


def test_disk_guard_spares_the_last_hour_then_takes_oldest_events(monkeypatch):
    cfg, db, events = setup(monkeypatch, free_gb=0)      # never enough space
    cfg.record.continuous_hours, cfg.record.event_days = 6, 10
    retention.prune(cfg, db, NOW)
    assert hours_left(cfg) == 2              # the current and the previous hour stay
    assert db.events_before() == []
