import asyncio
import tempfile
from pathlib import Path

from smartcam.alarm import ARMED, DISARMED, TRIGGERED, Alarm, AlarmConfig
from smartcam.config import load_config
from smartcam.db import Database
from smartcam.events import EventEngine, OpenEvent, RulesConfig
from smartcam.lights.dahua_rpc import SIREN, WARNING_LIGHT

TEST_CONFIG = Path(__file__).parent / "config.yml"


class FakeOutput:
    def __init__(self):
        self.calls = []

    async def output(self, kind, on):
        self.calls.append((kind, on))


def setup(**ac):
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    db = Database(cfg.paths.data / "t.db")
    engine = EventEngine(cfg, db, RulesConfig())
    out = FakeOutput()
    alarm = Alarm(cfg, db, engine, AlarmConfig(**{"light_minutes": 0.001, "siren_on_seconds": 0.01,
                                                  "siren_off_seconds": 0.01, **ac}), {"front_door": out})
    return alarm, engine, out


def event(score: float) -> OpenEvent:
    ev = OpenEvent(1, "living_room", "indoor", 0, 0)
    ev.score, ev.reasons = score, ["person", "indoor while away"]
    return ev


def test_arming_switches_away_on_and_disarming_off():
    async def go():
        alarm, engine, _ = setup()
        await alarm.arm()
        assert alarm.state == ARMED and engine.away
        await alarm.disarm()
        assert alarm.state == DISARMED and not engine.away
    asyncio.run(go())


def test_serious_event_while_armed_flashes_lights_then_goes_quiet_and_stays_spent():
    async def go():
        alarm, engine, out = setup()
        await alarm.arm()
        await alarm._scored(event(5))                 # below trigger_score: nothing
        assert alarm.state == ARMED and out.calls == []
        await alarm._scored(event(10))
        assert alarm.state == TRIGGERED
        await alarm._task
        assert out.calls[0] == (WARNING_LIGHT, True) and out.calls[-1] == (WARNING_LIGHT, False)
        assert all(k != SIREN for k, _ in out.calls)  # siren off by default
        assert alarm.state == ARMED                   # back to armed...
        out.calls.clear()
        await alarm._scored(event(10))                # ...but spent until disarmed and armed again
        assert alarm.state == ARMED and out.calls == []
        await alarm.disarm(); await alarm.arm()
        await alarm._scored(event(10))
        assert alarm.state == TRIGGERED
        await alarm.close()
    asyncio.run(go())


def test_siren_rounds_when_enabled():
    async def go():
        alarm, engine, out = setup(siren=True, siren_rounds=2)
        await alarm.arm()
        await alarm.trigger()
        await alarm._task
        assert out.calls.count((SIREN, True)) == 2 and out.calls.count((SIREN, False)) >= 2
    asyncio.run(go())


def test_disarm_during_alarm_switches_everything_off():
    async def go():
        alarm, engine, out = setup(light_minutes=10)
        await alarm.arm()
        await alarm.trigger()
        await asyncio.sleep(0.05)
        await alarm.disarm()
        await asyncio.sleep(0.05)
        assert out.calls[-1] == (WARNING_LIGHT, False) and alarm.state == DISARMED
    asyncio.run(go())


def test_not_armed_never_triggers():
    async def go():
        alarm, engine, out = setup()
        await alarm._scored(event(10))
        assert alarm.state == DISARMED and out.calls == []
    asyncio.run(go())
