import random
from datetime import date
from zoneinfo import ZoneInfo

from smartcam.presence import PresenceConfig, intervals, plan, sun_times, warp

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
    lib = [{"day": "2026-10-14", "weekend": False, "sun": (7.5 * 3600, 17.2 * 3600),
            "lights": {"light.kitchen": [(10 * 3600, 11 * 3600), (17.5 * 3600, 21 * 3600)]}},
           {"day": "2026-10-18", "weekend": True, "sun": (7.6 * 3600, 17.0 * 3600), "lights": {"light.kitchen": [(19 * 3600, 20 * 3600)]}}]
    src, actions = plan(date(2026, 10, 27), lib, PC, (8 * 3600, 16.6 * 3600), CPH, random.Random(1))   # a Tuesday
    assert src == "2026-10-14"
    assert len(actions) == 2 and actions[0].on and not actions[1].on   # the 10-11 daytime interval is skipped
    assert plan(date(2026, 10, 27), [], PC, (8 * 3600, 16.6 * 3600), CPH, random.Random(1)) == (None, [])
