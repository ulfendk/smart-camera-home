import asyncio
import io
import tempfile
import wave
from pathlib import Path

from smartcam.alarm import Alarm, AlarmConfig
from smartcam.config import load_config
from smartcam.db import Database
from smartcam.escalation import Escalation, EscalationConfig, siren
from smartcam.events import EventEngine, RulesConfig

TEST_CONFIG = Path(__file__).parent / "config.yml"


class FakeHa:
    def __init__(self):
        self.calls = []

    async def post(self, url, json):
        self.calls.append((url.removeprefix("/api/services/").replace("/", "."), json))
        return type("R", (), {"raise_for_status": lambda self: None})()

    async def get(self, url):
        states = [{"entity_id": "light.hall", "state": "off", "attributes": {}},
                  {"entity_id": "light.kitchen", "state": "on",
                   "attributes": {"brightness": 77, "color_mode": "color_temp", "color_temp_kelvin": 2700}},
                  {"entity_id": "light.gone", "state": "unavailable", "attributes": {}},
                  {"entity_id": "light.all_downstairs", "state": "on", "attributes": {"entity_id": ["light.kitchen"]}},
                  {"entity_id": "switch.x", "state": "on", "attributes": {}}]
        return type("R", (), {"raise_for_status": lambda self: None, "json": lambda self: states})()

    def services(self):
        return [c[0] for c in self.calls]


def setup(**ec):
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    cfg.ha.smartcam_host = "192.0.2.5"
    db = Database(cfg.paths.data / "t.db")
    alarm = Alarm(cfg, db, EventEngine(cfg, db, RulesConfig()), AlarmConfig(exit_delay=0, entry_delay=0), {})
    ha = FakeHa()
    esc = Escalation(cfg, alarm, EscalationConfig(enabled=True, speakers=["media_player.kitchen", "media_player.hall"], **ec), ha)
    return alarm, esc, ha


def test_the_siren_is_a_wav_of_half_a_minute():
    with wave.open(io.BytesIO(siren())) as w:
        assert w.getnchannels() == 1 and 29 < w.getnframes() / w.getframerate() < 31


def test_confirming_turns_every_light_up_and_plays_the_siren_then_stop_puts_it_back():
    async def go():
        alarm, esc, ha = setup()
        incident = await alarm.alert("intrusion", "living_room: indoor while away")
        await alarm.respond(f"SMARTCAM_CONFIRM_{incident['id']}", "Regin")
        await asyncio.sleep(0.05)
        ups = [c[1]["entity_id"] for c in ha.calls if c[0] == "light.turn_on"]
        assert ups == ["light.hall", "light.kitchen"]           # not unavailable lights, not groups
        assert all(c[1]["brightness_pct"] == 100 for c in ha.calls if c[0] == "light.turn_on")
        plays = [c[1] for c in ha.calls if c[0] == "media_player.play_media"]
        assert {p["entity_id"] for p in plays} == {"media_player.kitchen", "media_player.hall"}
        assert plays[0]["media_content_id"] == "http://192.0.2.5:8971/siren.wav"
        ha.calls.clear()
        await alarm.respond(f"SMARTCAM_STOP_{incident['id']}", "Regin")
        assert "sonos.restore" in ha.services() and not esc.active
        back = {c[1]["entity_id"]: c for c in ha.calls if c[0].startswith("light.")}
        assert back["light.hall"][0] == "light.turn_off"
        assert back["light.kitchen"] == ("light.turn_on", {"entity_id": "light.kitchen", "brightness": 77,
                                                           "color_temp_kelvin": 2700})
    asyncio.run(go())


def test_disarming_stops_it_and_a_false_alarm_never_starts_it():
    async def go():
        alarm, esc, ha = setup()
        await alarm.arm()
        incident = await alarm.alert("intrusion", "test")
        await alarm.respond(f"SMARTCAM_CONFIRM_{incident['id']}")
        assert esc.active
        await alarm.disarm()
        assert not esc.active and ha.services()[-1].startswith("light.")
        ha.calls.clear()
        incident = await alarm.alert("smoke", "smoke at kitchen")
        await alarm.respond(f"SMARTCAM_FALSE_{incident['id']}")
        assert not esc.active and ha.calls == []
    asyncio.run(go())


def test_the_confirmed_push_offers_to_stop_the_siren():
    from smartcam.ha.notify import response_payload
    cfg = load_config(TEST_CONFIG)
    incident = {"id": 4, "kind": "intrusion", "why": "x", "event_id": None, "status": "confirmed", "by": "Regin"}
    assert "actions" not in response_payload(cfg, incident)["data"]
    cfg.escalation = {"enabled": True}
    assert response_payload(cfg, incident)["data"]["actions"][0]["action"] == "SMARTCAM_STOP_4"
