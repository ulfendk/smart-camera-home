import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from smartcam.config import load_config
from smartcam.db import Database
from smartcam.events import EventEngine, RulesConfig
from smartcam.web import create_app

TEST_CONFIG = Path(__file__).parent / "config.yml"


def client() -> TestClient:
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    return TestClient(create_app(cfg, db, EventEngine(cfg, db, RulesConfig())))


def test_ingress_double_slash_serves_the_ui():
    c = client()
    assert c.get("/").status_code == 200
    assert c.get("//").status_code == 200          # how Home Assistant ingress asks for the entry page
    assert c.get("http://testserver//api/cameras").json()[0]["name"] == "front_door"


def test_go2rtc_proxy_only_exposes_the_player():
    c = client()
    assert c.get("/go2rtc/api/config").status_code == 404
    assert c.get("/go2rtc/api/streams").status_code == 404



def test_event_filters():
    import tempfile
    from smartcam.db import Database
    db = Database(Path(tempfile.mkdtemp()) / "t.db")
    p = db.insert_event(camera="front_door", label="person", start_ts=100, max_score=0.9)
    db.insert_event(camera="driveway", label="car", start_ts=200, max_score=0.8)
    db.update_event(p, notified=1)
    ids = lambda **kw: [e["id"] for e in db.events(**kw)]
    assert ids(kind="vehicle") == [2] and ids(kind="person") == [1] and ids(kind="notified") == [1]
    assert ids(camera="driveway") == [2]
    assert ids(since=150) == [2] and ids(until=150) == [1]



def test_event_download_is_an_attachment():
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    c = TestClient(create_app(cfg, db, EventEngine(cfg, db, RulesConfig())))
    (cfg.paths.data / "snapshots").mkdir(exist_ok=True)
    (cfg.paths.data / "snapshots" / "1.jpg").write_bytes(b"jpeg")
    ev = db.insert_event(camera="front_door", label="person", start_ts=1_700_000_000, max_score=0.9)
    db.update_event(ev, snapshot="snapshots/1.jpg")
    r = c.get(f"/api/events/{ev}/download/snapshot")
    assert r.status_code == 200 and r.content == b"jpeg"
    assert r.headers["content-disposition"].startswith('attachment; filename="smartcam_front_door_')
    assert c.get(f"/api/events/{ev}/download/clip").status_code == 404      # no clip yet
    assert c.get(f"/api/events/{ev}/download/db").status_code == 404        # only the event's media
    assert c.get("/api/events/99/download/snapshot").status_code == 404


def test_activity_lists_cameras_with_a_person_in_view():
    import time
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    engine = EventEngine(cfg, db, RulesConfig())
    c = TestClient(create_app(cfg, db, engine))
    assert c.get("/api/activity").json() == {"people": []}
    engine.people_seen.update(front_door=time.time(), driveway=time.time() - 60)
    assert c.get("/api/activity").json() == {"people": ["front_door"]}


def test_wall_still_is_a_jpeg():
    r = client().get("/api/wall.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"


def test_false_alarm_makes_its_spot_hot_and_taking_it_back_cools_it():
    from smartcam.static_map import StaticMap
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    engine = EventEngine(cfg, db, RulesConfig())
    engine.static_maps["front_door"] = sm = StaticMap()
    c = TestClient(create_app(cfg, db, engine))
    trunk = (0.62, 0.0, 0.81, 0.53)
    ev = db.insert_event(camera="front_door", label="person", start_ts=1_700_000_000, max_score=0.6)
    db.update_event(ev, box=list(trunk))
    assert c.post(f"/api/events/{ev}/review", json={"review": "false_positive"}).json()["box"] == list(trunk)
    assert sm.is_hot(trunk, 1_700_000_100)
    c.post(f"/api/events/{ev}/review", json={"review": "false_positive"})   # again: no extra heat
    c.post(f"/api/events/{ev}/review", json={"review": "ok"})
    assert not sm.is_hot(trunk, 1_700_000_100)
