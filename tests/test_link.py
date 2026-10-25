import asyncio
import tempfile
from pathlib import Path

from smartcam.alarm import Alarm, AlarmConfig
from smartcam.config import load_config
from smartcam.db import Database
from smartcam.events import EventEngine, RulesConfig
from smartcam.ha.link import HaLink, is_on, service_for

TEST_CONFIG = Path(__file__).parent / "config.yml"


def test_states_and_services():
    assert is_on("on") and is_on("armed_away") and not is_on("off") and not is_on("disarmed")
    assert is_on("unavailable") is None
    assert service_for("input_boolean.alarm", True) == ("input_boolean", "turn_on")
    assert service_for("alarm_control_panel.house", False) == ("alarm_control_panel", "alarm_disarm")


def test_follows_home_assistant_and_reports_back_without_echo():
    async def go():
        cfg = load_config(TEST_CONFIG)
        cfg.paths.data = Path(tempfile.mkdtemp())
        cfg.ha.alarm_entity = "input_boolean.alarm"
        db = Database(cfg.paths.data / "t.db")
        engine = EventEngine(cfg, db, RulesConfig())
        alarm = Alarm(cfg, db, engine, AlarmConfig(exit_delay=0, entry_delay=0), outputs={})
        link = HaLink(cfg, engine, alarm)
        sent = []

        class FakeWs:
            async def send(self, m): sent.append(m)
        link.ws = FakeWs()

        await link._apply("input_boolean.alarm", True)          # keypad armed it in HA
        assert alarm.state == "armed_away" and sent == []        # no echo back to HA
        await alarm.disarm()                                     # disarmed in SmartCam's UI
        assert len(sent) == 1 and '"turn_off"' in sent[0]
        await link._apply("input_boolean.alarm", False)          # HA confirms: nothing more to do
        assert len(sent) == 1 and alarm.state == "disarmed"
    asyncio.run(go())



def test_armed_home_is_not_mirrored_and_not_undone():
    async def go():
        cfg = load_config(TEST_CONFIG)
        cfg.paths.data = Path(tempfile.mkdtemp())
        cfg.ha.alarm_entity = "input_boolean.alarm"
        db = Database(cfg.paths.data / "t.db")
        engine = EventEngine(cfg, db, RulesConfig())
        alarm = Alarm(cfg, db, engine, AlarmConfig(exit_delay=0, entry_delay=0), outputs={})
        link = HaLink(cfg, engine, alarm)
        sent = []

        class FakeWs:
            async def send(self, m): sent.append(m)
        link.ws = FakeWs()
        await alarm.arm_home()
        assert sent == []                                   # the doors are not locked by "Alarm til"
        await link._apply("input_boolean.alarm", False)    # the boolean is off: armed home stays
        assert alarm.state == "armed_home"
        await link._apply("input_boolean.alarm", True)     # leaving via the keypad: away takes over
        assert alarm.state == "armed_away"
    asyncio.run(go())
