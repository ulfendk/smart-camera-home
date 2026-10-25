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
