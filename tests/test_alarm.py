import asyncio
import tempfile
from pathlib import Path

from smartcam.alarm import ARMED, DISARMED, TRIGGERED, Alarm, AlarmConfig
from smartcam.config import load_config
from smartcam.db import Database
from smartcam.events import EventEngine, OpenEvent, RulesConfig
from smartcam.lights.dahua_rpc import WARNING_LIGHT

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
    alarm = Alarm(cfg, db, engine, AlarmConfig(**{"light_minutes": 0.001, "exit_delay": 0, "entry_delay": 0, **ac}),
                  {"front_door": out})
    return alarm, engine, out


def event(score: float, start: float | None = None) -> OpenEvent:
    import time
    start = time.time() if start is None else start
    ev = OpenEvent(1, "living_room", "indoor", start, start)
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
        assert alarm.state == ARMED                   # back to armed...
        out.calls.clear()
        await alarm._scored(event(10))                # ...but spent until disarmed and armed again
        assert alarm.state == ARMED and out.calls == []
        await alarm.disarm(); await alarm.arm()
        await alarm._scored(event(10))
        assert alarm.state == TRIGGERED
        await alarm.close()
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


def test_exit_delay_and_people_already_inside_do_not_trigger():
    async def go():
        import time
        alarm, engine, out = setup(exit_delay=0.2)
        inside = event(10, start=time.time())          # someone moving about while arming
        await alarm.arm()
        assert alarm.state == "arming" and not engine.away
        await alarm._scored(event(10))
        assert alarm.state == "arming"                  # nothing counts during the exit delay
        await asyncio.sleep(0.3)
        assert alarm.state == ARMED and engine.away
        await alarm._scored(inside)                     # started before the alarm was active
        assert alarm.state == ARMED and out.calls == []
        await alarm._scored(event(10))                  # new activity after arming
        assert alarm.state == TRIGGERED
        await alarm.close()
    asyncio.run(go())


def test_entry_delay_lets_you_disarm_before_anything_happens():
    async def go():
        alarm, engine, out = setup(entry_delay=0.3)
        await alarm.arm()
        await alarm._scored(event(10))
        assert alarm.state == "pending" and out.calls == []
        await alarm.disarm()
        await asyncio.sleep(0.4)
        assert alarm.state == DISARMED and out.calls == []   # no lights, no alarm push
    asyncio.run(go())


def test_armed_home_never_goes_off_and_switches_cleanly():
    async def go():
        alarm, engine, out = setup()
        await alarm.command("ARM_HOME")
        assert alarm.state == "armed_home" and engine.home_armed and not engine.away
        await alarm._scored(event(10))                      # nothing triggers at home
        assert alarm.state == "armed_home" and out.calls == []
        await alarm.command("ARM_AWAY")                     # leaving: away takes over
        assert alarm.state == ARMED and not engine.home_armed and engine.away
        await alarm.disarm()
        assert alarm.state == DISARMED and not engine.home_armed and not engine.away
    asyncio.run(go())


def test_morning_person_on_the_stairs_disarms_armed_home(monkeypatch):
    async def go():
        alarm, engine, out = setup(home_disarm_camera="living_room", home_disarm_after="05:00")
        await alarm.arm_home()
        monkeypatch.setattr(alarm, "_local_minutes", lambda: 4 * 60 + 30)    # 04:30: too early
        await alarm._occupancy("living_room", True)
        assert alarm.state == "armed_home"
        monkeypatch.setattr(alarm, "_local_minutes", lambda: 6 * 60 + 15)    # 06:15
        await alarm._occupancy("front_door", True)                           # another camera: no
        assert alarm.state == "armed_home"
        await alarm._occupancy("living_room", True)
        assert alarm.state == "disarmed" and not engine.home_armed
    asyncio.run(go())


def test_confirming_an_alarm_announces_it_and_keeps_the_lights():
    async def go():
        alarm, engine, out = setup(light_minutes=1)
        answers = []
        alarm.on_response.append(lambda inc: asyncio.sleep(0, answers.append(inc)))
        await alarm.arm()
        await alarm.trigger("test", delay=0)
        n = alarm.incident["id"]
        assert alarm.incident["status"] == "open" and alarm.incident["kind"] == "intrusion"
        await alarm.respond(f"SMARTCAM_CONFIRM_{n}", "Regin")
        assert alarm.incident["status"] == "confirmed" and alarm.incident["by"] == "Regin"
        assert alarm.state == TRIGGERED and answers[0]["status"] == "confirmed"
        await alarm.respond(f"SMARTCAM_FALSE_{n}", "someone")   # answered already: no change
        assert alarm.incident["status"] == "confirmed" and len(answers) == 1
        await alarm.close()
    asyncio.run(go())


def test_false_alarm_switches_the_lights_off_marks_the_event_and_stays_armed():
    async def go():
        alarm, engine, out = setup(light_minutes=1)
        ev = alarm.db.insert_event(camera="living_room", label="person", start_ts=1.0, max_score=0.6)
        await alarm.arm()
        await alarm.trigger("test", event_id=ev, delay=0)
        await asyncio.sleep(0)
        await alarm.respond(f"SMARTCAM_FALSE_{alarm.incident['id']}")
        try:
            await alarm._task
        except asyncio.CancelledError:
            pass
        assert alarm.state == ARMED and out.calls[-1] == (WARNING_LIGHT, False)
        assert alarm.db.event(ev)["review"] == "false_positive"
    asyncio.run(go())


def test_an_old_push_cannot_answer_a_newer_alarm():
    async def go():
        alarm, engine, out = setup()
        await alarm.alert("smoke", "smoke at kitchen")
        await alarm.alert("smoke", "smoke at hall")
        assert await alarm.respond("SMARTCAM_CONFIRM_1") is None
        assert alarm.incident["status"] == "open"
        assert (await alarm.respond("SMARTCAM_CONFIRM_2"))["status"] == "confirmed"
    asyncio.run(go())
