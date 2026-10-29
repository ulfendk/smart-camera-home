import asyncio
import tempfile
from pathlib import Path

import numpy as np

from smartcam import smoke as smoke_mod
from smartcam.alarm import Alarm, AlarmConfig
from smartcam.config import load_config
from smartcam.db import Database
from smartcam.events import EventEngine, RulesConfig
from smartcam.ha.link import HaLink
from smartcam.smoke import Smoke, SmokeConfig

TEST_CONFIG = Path(__file__).parent / "config.yml"


def setup(monkeypatch):
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    engine = EventEngine(cfg, db, RulesConfig())
    alarm = Alarm(cfg, db, engine, AlarmConfig(), {})
    grabbed = []

    async def grab(cfg, http, camera):
        grabbed.append(camera)
        return np.full((90, 160, 3), 128, np.uint8)
    monkeypatch.setattr(smoke_mod, "grab", grab)
    smoke = Smoke(cfg, db, alarm, None, SmokeConfig(entities=["binary_sensor.kitchen_smoke"], height=90))
    return smoke, alarm, db, grabbed


def test_smoke_pushes_a_picture_from_inside_as_an_alarm_incident(monkeypatch):
    async def go():
        smoke, alarm, db, grabbed = setup(monkeypatch)
        pushed = []
        alarm.on_trigger.append(lambda inc, ev: asyncio.sleep(0, pushed.append((inc, ev))))
        inc = await smoke.alarm_for("binary_sensor.kitchen_smoke", "Kitchen smoke")
        assert grabbed == ["living_room"]                  # the indoor cameras by default
        assert inc["kind"] == "smoke" and "Kitchen smoke" in inc["why"]
        ev = pushed[0][1]
        assert ev["label"] == "smoke" and ev["snapshot"]
        assert (smoke.cfg.paths.data / ev["snapshot"]).exists()
        assert await smoke.alarm_for("binary_sensor.kitchen_smoke", "Kitchen smoke") is None   # cooldown
        assert len(pushed) == 1
    asyncio.run(go())


def test_ha_link_hands_smoke_and_push_answers_on(monkeypatch):
    async def go():
        smoke, alarm, db, _ = setup(monkeypatch)
        link = HaLink(smoke.cfg, alarm.engine, alarm, smoke)
        link._person({"entity_id": "person.regin", "attributes": {"user_id": "u1", "friendly_name": "Regin"}})
        await link._event({"event_type": "state_changed", "data": {
            "entity_id": "binary_sensor.kitchen_smoke", "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen smoke"}}}})
        await asyncio.sleep(0.05)
        n = alarm.incident["id"]
        await link._event({"event_type": "mobile_app_notification_action", "data": {"action": f"SMARTCAM_CONFIRM_{n}"},
                           "context": {"user_id": "u1"}})
        assert alarm.incident["status"] == "confirmed" and alarm.incident["by"] == "Regin"
    asyncio.run(go())
