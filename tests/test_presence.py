import random
from datetime import date
from zoneinfo import ZoneInfo

from datetime import datetime

from smartcam.presence import PresenceConfig, at, clock_of, holiday_set, intervals, kind, plan, sun_times, warp

CPH = ZoneInfo("Europe/Copenhagen")
PC = PresenceConfig()


def hm(s: float) -> str:
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}"


def test_sun_times_are_about_right_for_copenhagen():
    rise, sset = sun_times(date(2026, 10, 26), 55.68, 12.57, CPH)
    assert "06:55" <= hm(rise) <= "07:10" and "16:35" <= hm(sset) <= "16:55"     # just after the clocks went back
    rise, sset = sun_times(date(2026, 6, 21), 55.68, 12.57, CPH)
    assert "04:20" <= hm(rise) <= "04:35" and "21:50" <= hm(sset) <= "22:05"


def test_warp_follows_sunset_but_keeps_bedtime():
    src, dst = (7 * 3600, 17 * 3600), (8 * 3600, 16.5 * 3600)    # recorded day vs. today
    bed = 23 * 3600
    assert warp(17 * 3600 + 1200, src, dst, bed) == 16.5 * 3600 + 1200 * (bed - 16.5 * 3600) / (bed - 17 * 3600)
    assert abs(warp(17 * 3600, src, dst, bed) - 16.5 * 3600) < 1     # dusk -> today's dusk
    assert warp(bed, src, dst, bed) == bed                          # bedtime stays
    assert warp(43200, src, dst, bed) == 43200


def test_motion_flicker_is_merged_and_blips_dropped():
    m = 0.0
    changes = [(m + 100, True), (m + 160, False), (m + 200, True), (m + 400, False),   # flicker: one visit
               (m + 5000, True), (m + 5020, False)]                                    # 20 s blip: dropped
    assert intervals(changes, m, m + 86400, PC) == [(100, 400)]


def test_plan_replays_a_same_kind_evening_and_skips_daylight():
    lib = [{"day": "2026-10-13", "weekend": False, "sun": (7.5 * 3600, 17.2 * 3600),
            "lights": {"light.kitchen": [(10 * 3600, 11 * 3600), (17.5 * 3600, 21 * 3600)]}},
           {"day": "2026-10-18", "weekend": True, "sun": (7.6 * 3600, 17.0 * 3600), "lights": {"light.kitchen": [(19 * 3600, 20 * 3600)]}}]
    src, actions = plan(date(2026, 10, 27), lib, PC, (8 * 3600, 16.6 * 3600), CPH, random.Random(1))   # a Tuesday
    assert src == "2026-10-13"                                  # a Tuesday for a Tuesday
    assert len(actions) == 2 and actions[0].on and not actions[1].on   # the 10-11 daytime interval is skipped
    assert plan(date(2026, 10, 27), [], PC, (8 * 3600, 16.6 * 3600), CPH, random.Random(1)) == (None, [])


def test_clock_times_survive_the_day_the_clocks_go_back():
    day = date(2026, 10, 25)                     # 25 hours long in Copenhagen
    ts = at(day, 7 * 3600, CPH)
    assert datetime.fromtimestamp(ts, CPH).strftime("%H:%M") == "07:00"
    assert clock_of(ts, day, CPH) == 7 * 3600
    assert clock_of(at(day + __import__("datetime").timedelta(days=1), 0, CPH), day, CPH) == 86400


def test_day_kinds_and_holidays_replay_the_latest_sunday():
    hols = holiday_set("DK", ["12-24"], range(2026, 2027))
    assert kind(date(2026, 12, 25), hols) == "sunday"          # Juledag, a Friday
    assert kind(date(2026, 12, 24), hols) == "sunday"          # extra holiday
    assert kind(date(2026, 10, 30), hols) == "friday" and kind(date(2026, 10, 31), hols) == "saturday"
    lib = [{"day": d, "weekend": True, "sun": (8 * 3600, 16 * 3600), "lights": {"light.x": [(18 * 3600, 20 * 3600)]}}
           for d in ("2026-12-06", "2026-12-20", "2026-12-13", "2026-12-19")]   # three Sundays, a Saturday
    src, _ = plan(date(2026, 12, 25), lib, PC, (8.5 * 3600, 15.7 * 3600), CPH, random.Random(3), hols)
    assert src == "2026-12-20"                                  # the latest Sunday
    src, _ = plan(date(2026, 12, 26), lib, PC, (8.5 * 3600, 15.7 * 3600), CPH, random.Random(3), set())
    assert src == "2026-12-19"                                  # a Saturday replays the Saturday


def test_brightness_is_learned_and_replayed():
    m = 0.0
    changes = [(m + 1000, True, 80), (m + 1600, True, 30), (m + 4000, True, 30), (m + 5000, False, None)]
    (iv,) = intervals(changes, m, m + 86400, PC)
    assert iv == (1000, 5000, 30)                       # 30% for most of the visit
    lib = [{"day": "2026-10-13", "weekend": False, "sun": (7.5 * 3600, 17.2 * 3600),
            "lights": {"light.kitchen": [[18 * 3600, 20 * 3600, 30]]}}]
    _, actions = plan(date(2026, 10, 27), lib, PC, (8 * 3600, 16.6 * 3600), CPH, random.Random(1))
    assert actions[0].brightness == 30 and actions[1].brightness is None


def test_own_recordings_are_kept_and_read_back():
    import tempfile
    from pathlib import Path
    from smartcam.db import Database
    from smartcam.presence import Library
    lib = Library(Database(Path(tempfile.mkdtemp()) / "t.db"))
    lib.record(100.0, "light.kitchen", True, 40)
    lib.record(200.0, "light.kitchen", False, None)
    assert lib.recorded(0, 1000) == {"light.kitchen": [(100.0, True, 40), (200.0, False, None)]}
    assert lib.recording_since() is None


def test_tv_counts_as_on_unless_off_or_standby():
    from smartcam.presence import is_on
    assert [is_on("media_player.tv", s) for s in ("playing", "paused", "idle", "off", "standby", "unavailable")] \
        == [True, True, True, False, False, False]
    assert is_on("light.x", "on") and is_on("light.x", "off") is False and is_on("light.x", "unavailable") is None
    pc = PresenceConfig(lights=["light.x"], tv="media_player.tv")
    lib = [{"day": "2026-10-13", "weekend": False, "sun": (7.5 * 3600, 17.2 * 3600),
            "lights": {"light.x": [(18 * 3600, 20 * 3600)], "media_player.tv": [(19 * 3600, 22 * 3600)],
                       "light.gone": [(18 * 3600, 19 * 3600)]}}]
    _, actions = plan(date(2026, 10, 27), lib, pc, (8 * 3600, 16.6 * 3600), CPH, random.Random(1))
    assert {a.light for a in actions} == {"light.x", "media_player.tv"}
    _, actions = plan(date(2026, 10, 27), lib, PresenceConfig(lights=["light.x"]), (8 * 3600, 16.6 * 3600), CPH, random.Random(1))
    assert {a.light for a in actions} == {"light.x"}


class FakeHa:
    def __init__(self, states):
        self.states, self.calls = states, []

    async def get(self, path):
        s = self.states.get(path.rsplit("/", 1)[1])
        return type("R", (), {"status_code": 200 if s else 404, "json": lambda self=None: {"state": s}})()

    async def post(self, path, json):
        domain, service = path.split("/")[-2:]
        self.calls.append((f"{domain}.{service}", json))
        if domain == "remote" and service == "turn_on":
            self.states["media_player.tv"] = "on"      # the TV set follows over HDMI-CEC
        elif service == "turn_off":
            self.states[json["entity_id"]] = "off"


def simulator(states):
    import tempfile
    from pathlib import Path
    from smartcam.db import Database
    from smartcam.presence import PresenceSimulator
    cfg = type("C", (), {"presence": {"lights": ["light.x"], "tv": "media_player.tv", "tv_remote": "remote.tv",
                                      "tv_app": "dk.dr.drtv.webos2", "tv_keys": ["RIGHT", "ENTER", "ENTER"]},
                         "ha": type("H", (), {"url": "http://ha", "token": "t"})()})()
    db = Database(Path(tempfile.mkdtemp()) / "t.db")
    sim = PresenceSimulator(cfg, db, alarm=None)
    sim.http = FakeHa(states)
    return sim, db


def test_tv_is_woken_and_the_app_reopened_and_driven_then_switched_off(monkeypatch):
    import asyncio
    import smartcam.presence as presence
    real_sleep = asyncio.sleep
    monkeypatch.setattr(presence.asyncio, "sleep", lambda s: real_sleep(0))   # no waiting for the TV to wake
    async def go():
        sim, _ = simulator({"media_player.tv": "off"})
        await sim._switch("media_player.tv", True)
        calls = sim.http.calls
        assert [c[0] for c in calls] == ["remote.turn_on", "webostv.command", "webostv.command"] + ["webostv.button"] * 3
        assert [c[1]["command"] for c in calls[1:3]] == ["system.launcher/close", "system.launcher/launch"]
        assert calls[2][1]["payload"] == {"id": "dk.dr.drtv.webos2"}
        assert [c[1]["button"] for c in calls[3:]] == ["RIGHT", "ENTER", "ENTER"]
        await sim._stop()
        assert sim.http.calls[-2:] == [("media_player.turn_off", {"entity_id": "media_player.tv"}),
                                       ("remote.turn_off", {"entity_id": "remote.tv"})]
        sim.http.calls.clear()
        sim.http.states["media_player.tv"] = "on"           # already on: nothing to do
        await sim._switch("media_player.tv", True)
        assert sim.http.calls == []
    asyncio.run(go())


def test_tv_switch_is_on_by_default_kept_and_turns_our_tv_off(monkeypatch):
    import asyncio
    import smartcam.presence as presence
    real_sleep = asyncio.sleep
    monkeypatch.setattr(presence.asyncio, "sleep", lambda s: real_sleep(0))   # no waiting for the TV to wake
    async def go():
        sim, db = simulator({"media_player.tv": "off"})
        seen = []
        async def cb(on):
            seen.append(on)
        sim.on_tv.append(cb)
        assert sim.tv_enabled
        await sim._switch("media_player.tv", True)
        await sim.set_tv(False)
        assert sim.http.states["media_player.tv"] == "off" and seen == [False]
        sim.http.calls.clear()
        await sim._switch("media_player.tv", True)          # left out while switched off
        assert sim.http.calls == []
        assert not presence.PresenceSimulator(sim.cfg, db, alarm=None).tv_enabled   # survives a restart
    asyncio.run(go())

