from pathlib import Path

from smartcam.config import load_config
from smartcam.power import PowerWatchdog, WatchdogConfig

TEST_CONFIG = Path(__file__).parent / "config.yml"
WD = WatchdogConfig(stale_seconds=120, cooldown_minutes=30, max_per_day=2)


def watchdog(ha: bool = True) -> PowerWatchdog:
    cfg = load_config(TEST_CONFIG)
    cfg.ha.enabled = ha
    cfg.cameras["living_room"].power_switch = "switch.plug"
    return PowerWatchdog(cfg, WD, now=0)


def test_switch_cameras_are_power_cycled_and_dahuas_rebooted():
    w = watchdog()
    assert list(w.switches) == ["living_room"] and list(w.dahua) == ["front_door", "driveway"]


def test_switches_need_home_assistant():
    assert watchdog(ha=False).switches == {}


def test_dead_stream_is_due_after_stale_seconds():
    w = watchdog()
    w.seen("living_room", now=100)
    assert "living_room" not in w.due(now=200)
    assert "living_room" in w.due(now=221)


def test_cooldown_and_daily_limit():
    w = watchdog()
    w.cycles["living_room"].append(200)       # cycled at t=200, camera never came back
    assert "living_room" not in w.due(now=200 + 29 * 60)    # cooling down
    assert "living_room" in w.due(now=200 + 31 * 60)
    w.cycles["living_room"].append(200 + 31 * 60)
    assert "living_room" not in w.due(now=200 + 120 * 60)   # 2 cycles today: the limit
    assert "living_room" in w.due(now=200 + 86400 + 1)
