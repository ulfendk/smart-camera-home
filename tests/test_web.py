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
