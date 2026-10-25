from pathlib import Path

from smartcam.config import load_config
from smartcam.power import PowerWatchdog, WatchdogConfig

TEST_CONFIG = Path(__file__).parent / "config.yml"
WD = WatchdogConfig(stale_seconds=120, cooldown_minutes=30, max_per_day=2)


def watchdog() -> PowerWatchdog:
    cfg = load_config(TEST_CONFIG)
    cfg.cameras["living_room"].power_switch = "switch.plug"
    return PowerWatchdog(cfg, WD, now=0)


def test_only_cameras_with_a_switch_are_watched():
    assert list(watchdog().switches) == ["living_room"]


def test_dead_stream_is_due_after_stale_seconds():
    w = watchdog()
    w.seen("living_room", now=100)
    assert w.due(now=200) == []
    assert w.due(now=221) == ["living_room"]


def test_cooldown_and_daily_limit():
    w = watchdog()
    w.cycles["living_room"].append(200)       # cycled at t=200, camera never came back
    assert w.due(now=200 + 29 * 60) == []    # cooling down
    assert w.due(now=200 + 31 * 60) == ["living_room"]
    w.cycles["living_room"].append(200 + 31 * 60)
    assert w.due(now=200 + 120 * 60) == []   # 2 cycles today: the limit
    assert w.due(now=200 + 86400 + 1) == ["living_room"]
