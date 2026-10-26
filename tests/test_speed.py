from smartcam.detect import Detection
from smartcam.speed import SpeedConfig, SpeedMeter, crossing
from smartcam.tracker import Track, TrackerConfig

# Two vertical lines 10 m apart across a horizontal road; the car drives left to right.
SC = SpeedConfig(line_a=[0.3, 0.4, 0.3, 0.9], line_b=[0.7, 0.4, 0.7, 0.9], distance_m=10)


def drive(kmh: float, fps: float = 5, label: str = "car", reverse: bool = False) -> list:
    """Feed a meter frames of a car crossing the scene at `kmh` (the lines are 0.4 of the width apart)."""
    meter, t = SpeedMeter("street", SC), Track(1, 0.0, TrackerConfig())
    width_per_s = 0.4 / (10 / (kmh / 3.6))       # normalised width covered per second
    readings, x, ts = [], 0.05, 0.0
    while 0 <= x <= 0.95:
        cx = (1 - x) if reverse else x
        t.add(ts, Detection(label, 0.8, (cx - 0.05, 0.55, cx + 0.05, 0.7)), None)
        readings += meter.update([t])
        ts += 1 / fps
        x += width_per_s / fps
    return readings


def test_crossing_is_interpolated_between_frames():
    assert abs(crossing((0.2, 0.6), (0.4, 0.6), SC.line_a) - 0.5) < 1e-9
    assert crossing((0.4, 0.6), (0.5, 0.6), SC.line_a) is None


def test_speed_is_measured_accurately_at_5_fps():
    for kmh in (20, 50, 80):
        (r,) = drive(kmh)
        assert abs(r.kmh - kmh) / kmh < 0.03 and r.direction == "a→b"


def test_either_direction_and_only_vehicles():
    (r,) = drive(40, reverse=True)
    assert abs(r.kmh - 40) < 1.5 and r.direction == "b→a"
    assert drive(40, label="person") == []


def test_not_ready_without_calibration():
    assert not SpeedConfig().ready and SC.ready


def test_best_shot_finds_the_vehicle_where_it_was_measured(monkeypatch, tmp_path):
    """The recording runs late or early against detection: pick the frame where the car is at the measured box."""
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from smartcam import events
    from smartcam.config import load_config
    from smartcam.db import Database
    from smartcam.detect import Detection

    cfg = load_config(Path(__file__).parent / "config.yml")
    engine = events.EventEngine(cfg, Database(tmp_path / "t.db"))
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    monkeypatch.setattr(events, "frames_between", lambda *a, **k: iter(enumerate(frames)))
    # The car drives right to left; it is at the measured spot in frame 1 only. A person and a parked car are ignored.
    boxes = {0: [Detection("car", 0.9, (0.6, 0.3, 0.8, 0.5))],
             1: [Detection("car", 0.9, (0.41, 0.3, 0.61, 0.5)), Detection("person", 0.9, (0.4, 0.3, 0.6, 0.5))],
             2: [Detection("car", 0.9, (0.1, 0.3, 0.3, 0.5)), Detection("car", 0.9, (0.0, 0.8, 0.1, 0.9))]}
    engine.detector = SimpleNamespace(detect=lambda img: boxes[int(img[0, 0, 0])])
    img, box = engine._find_vehicle(SimpleNamespace(camera="front_door", box=(0.4, 0.3, 0.6, 0.5)), 100.0)
    assert img[0, 0, 0] == 1 and box == (0.41, 0.3, 0.61, 0.5)

    engine.detector = None
    assert engine._find_vehicle(SimpleNamespace(camera="front_door", box=(0.4, 0.3, 0.6, 0.5)), 100.0) is None
