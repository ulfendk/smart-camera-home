import numpy as np

from smartcam.detect import Detection
from smartcam.motion import MotionResult
from smartcam.tracker import Tracker

CAR = (0.40, 0.40, 0.70, 0.70)


def motion_over(*boxes) -> MotionResult:
    mask = np.zeros((180, 320), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[int(y1 * 180):int(y2 * 180), int(x1 * 320):int(x2 * 320)] = 255
    return MotionResult(mask, 0.1, list(boxes))


def person_at(x: float) -> tuple:
    return (x, 0.35, x + 0.08, 0.75)


def test_parked_car_becomes_stationary_and_never_moving():
    tr = Tracker()
    for i in range(40 * 5):  # 40 s at 5 fps
        tr.update(i / 5, [Detection("car", 0.8, CAR)])
    car = next(iter(tr.tracks.values()))
    assert car.stationary and not car.moving


def test_person_walking_past_parked_car_moves_car_does_not():
    tr = Tracker()
    for i in range(40 * 5):
        tr.update(i / 5, [Detection("car", 0.8, CAR)])
    for i in range(25):  # person walks across, in front of the car
        x = 0.30 + i * 0.02
        p = person_at(x)
        tr.update(40 + i / 5, [Detection("car", 0.8, CAR), Detection("person", 0.7, p)], motion_over(p))
    tracks = {t.label: t for t in tr.tracks.values()}
    assert tracks["person"].moving
    assert tracks["car"].stationary and not tracks["car"].moving


def test_lamp_detected_as_person_with_motion_nearby_stays_stationary():
    tr = Tracker()
    lamp = (0.90, 0.47, 0.95, 0.58)
    for i in range(35 * 5):
        tr.update(i / 5, [Detection("person", 0.55, lamp)])
    # Someone walks right past the lamp: lots of motion inside its box.
    for i in range(10):
        tr.update(35 + i / 5, [Detection("person", 0.55, lamp)], motion_over((0.85, 0.3, 1.0, 0.9)))
    t = next(iter(tr.tracks.values()))
    assert t.stationary and not t.moving


def test_person_moving_in_place_is_active_before_stationary():
    tr = Tracker()
    p = person_at(0.5)
    for i in range(10):
        tr.update(i / 5, [Detection("person", 0.7, p)], motion_over(p))
    assert next(iter(tr.tracks.values())).moving


def test_fast_mover_without_overlap_keeps_one_track():
    tr = Tracker()
    for i in range(10):
        tr.update(i / 5, [Detection("person", 0.7, person_at(0.1 + i * 0.07))])
    assert len(tr.tracks) == 1


def test_weak_detection_cannot_start_track_but_extends_one():
    tr = Tracker()
    tr.update(0, [Detection("person", 0.35, person_at(0.2))])
    assert not tr.tracks
    tr.update(0.2, [Detection("person", 0.7, person_at(0.2))])
    tr.update(0.4, [Detection("person", 0.35, person_at(0.21))])
    assert next(iter(tr.tracks.values())).hits == 2


def test_track_ends_after_max_age():
    tr = Tracker()
    tr.update(0, [Detection("person", 0.7, person_at(0.2))])
    _, ended = tr.update(5, [])
    assert len(ended) == 1 and not tr.tracks
