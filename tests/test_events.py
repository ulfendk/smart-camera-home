from datetime import datetime

from smartcam.events import OpenEvent, RulesConfig, is_night, score_event, should_notify

RULES = RulesConfig(door_zones=["door"])
NOON = datetime(2026, 10, 25, 12, 0).timestamp()
MIDNIGHT = datetime(2026, 10, 25, 23, 30).timestamp()


def ev(location="outdoor", **dwell) -> OpenEvent:
    return OpenEvent(1, "front_door", location, NOON, NOON, zone_dwell=dwell)


def scored(e: OpenEvent, now: float, away: bool, others=()) -> OpenEvent:
    e.score, e.reasons = score_event(e, now, away, RULES, list(others))
    return e


def test_night_wraps_midnight():
    assert is_night(MIDNIGHT, RULES) and not is_night(NOON, RULES)
    assert is_night(datetime(2026, 10, 25, 3).timestamp(), RULES)


def test_daytime_passerby_at_home_is_not_notified():
    e = scored(ev(), NOON, away=False)
    assert e.score == 1 and not should_notify(e, False, NOON, RULES)


def test_any_outdoor_person_while_away_is_notified():
    e = scored(ev(), NOON, away=True)
    assert e.score == 4 and should_notify(e, True, NOON, RULES)


def test_indoor_person_while_away_is_max():
    e = scored(ev("indoor"), NOON, away=True)
    assert e.score == 10 and should_notify(e, True, NOON, RULES)


def test_indoor_family_at_home_never_notified():
    e = scored(ev("indoor"), MIDNIGHT, away=False)
    assert not should_notify(e, False, MIDNIGHT, RULES)


def test_night_loiterer_at_door_while_home_is_notified():
    e = scored(ev(door=45), MIDNIGHT, away=False)
    assert e.score == 5  # person + night + loitering + door
    assert should_notify(e, False, MIDNIGHT, RULES)


def test_quick_night_passerby_while_home_is_not_notified():
    e = scored(ev(), MIDNIGHT, away=False)
    assert e.score == 2 and not should_notify(e, False, MIDNIGHT, RULES)


def test_armed_home_warns_only_about_very_suspicious_outdoor_activity():
    home = lambda e, t: (score_event(e, t, False, RULES, [], home_armed=True))
    e = ev(); e.score, e.reasons = home(e, MIDNIGHT)
    assert e.score == 3 and not should_notify(e, False, MIDNIGHT, RULES, home_armed=True)   # just passing at night
    e = ev(door=45); e.score, e.reasons = home(e, MIDNIGHT)
    assert e.score == 6 and should_notify(e, False, MIDNIGHT, RULES, home_armed=True)      # loitering at the door
    e = ev(); e.score, e.reasons = score_event(e, MIDNIGHT, False, RULES, ["driveway"], home_armed=True)
    assert e.score == 4 and should_notify(e, False, MIDNIGHT, RULES, home_armed=True)      # camera to camera
    e = ev("indoor"); e.score, e.reasons = home(e, MIDNIGHT)
    assert e.score == 2 and not should_notify(e, False, MIDNIGHT, RULES, home_armed=True)


def test_cross_camera_sequence_adds_suspicion():
    e = scored(ev(), MIDNIGHT, away=False, others=["driveway"])
    assert "seen on driveway just before" in e.reasons and e.score == 3


# --- event lifecycle -------------------------------------------------------

import asyncio
import tempfile
from pathlib import Path

import numpy as np

from smartcam.config import load_config

TEST_CONFIG = Path(__file__).parent / "config.yml"
from smartcam.db import Database
from smartcam.detect import Detection
from smartcam.events import EventEngine
from smartcam.ingest import Frame
from smartcam.motion import MotionResult
from smartcam.pipeline import FrameResult
from smartcam.tracker import Track, TrackerConfig


def make_engine(merge_gap: float):
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    return EventEngine(cfg, db, RulesConfig(merge_gap=merge_gap)), db


def track(tid: int, ts: float) -> Track:
    t = Track(tid, ts, TrackerConfig())
    t.add(ts, Detection("person", 0.8, (0.4, 0.3, 0.5, 0.8)), None)
    return t


def res(ts, confirmed=(), tracks=(), ended=()) -> FrameResult:
    frame = Frame("living_room", ts, np.zeros((36, 64, 3), np.uint8))
    return FrameResult(frame, MotionResult(np.zeros((9, 16), np.uint8), 0), [], list(tracks),
                       list(confirmed), list(ended))


def test_reappearing_person_continues_the_event_and_closes_after_gap():
    async def go():
        # The gap is measured from the last sighting (track.last_ts), not from when the tracker gives up.
        engine, db = make_engine(merge_gap=0.5)
        closed = []
        engine.on_close.append(lambda e: asyncio.sleep(0, closed.append(e["id"])))
        a = track(1, 0.0)                               # last seen at 0.0
        await engine.handle(res(0.0, confirmed=[a], tracks=[a]))
        await engine.handle(res(0.1, ended=[a]))
        b = track(2, 0.3)                               # same person picked up again within the gap
        await engine.handle(res(0.3, confirmed=[b], tracks=[b]))
        await asyncio.sleep(0.6)                        # longer than merge_gap, but b is still alive
        assert len(db.events()) == 1 and closed == []
        await engine.handle(res(1.0, ended=[b]))
        await asyncio.sleep(0.6)
        assert closed == [db.events()[0]["id"]]
        c = track(3, 5.0)                               # long after: a new event
        await engine.handle(res(5.0, confirmed=[c], tracks=[c]))
        assert len(db.events()) == 2
    asyncio.run(go())
