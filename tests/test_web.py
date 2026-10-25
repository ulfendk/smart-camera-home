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
