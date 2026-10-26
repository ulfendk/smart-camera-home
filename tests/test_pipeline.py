import numpy as np

from smartcam.config import CameraConfig
from smartcam.detect import Detection
from smartcam.ingest import Frame
from smartcam.motion import MotionResult
from smartcam.pipeline import CameraPipeline
from smartcam.static_map import StaticMap

CAM = CameraConfig(name="test", location="indoor", ip="0.0.0.0", record="x", detect="x")
LAMP = (0.90, 0.47, 0.95, 0.58)


class Scripted:
    """Detector stub returning pre-planned detections per call."""

    def __init__(self, plan):
        self.plan = iter(plan)

    def detect(self, image):
        return next(self.plan, [])


def frame(ts: float) -> Frame:
    return Frame("test", ts, np.full((360, 640, 3), 100, np.uint8))


def run(pipeline: CameraPipeline, n: int, t0: float = 0.0):
    out = []
    for i in range(n):
        out.extend(pipeline.process(frame(t0 + i / 5)).confirmed)
    return out


def test_walking_person_confirmed_once():
    plan = [[Detection("person", 0.8, (0.1 + i * 0.03, 0.3, 0.18 + i * 0.03, 0.8))] for i in range(15)]
    p = CameraPipeline(CAM, Scripted(plan))
    p.cfg.detect_refresh = 0  # no motion in the blank frames; detect every frame
    confirmed = run(p, 15)
    assert len(confirmed) == 1 and confirmed[0].label == "person"


def test_static_lamp_never_confirmed_and_heats_map():
    sm = StaticMap(hot_threshold=1)
    plan = [[Detection("person", 0.6, LAMP)] for _ in range(100)]
    p = CameraPipeline(CAM, Scripted(plan), sm)
    p.cfg.detect_refresh = 0
    assert run(p, 100) == []
    run(p, 25, t0=30)  # no more detections: the lamp track ends
    assert sm.is_hot(LAMP, 40)


class Moving:
    """Motion stub: the whole frame is in motion."""

    def update(self, image):
        return MotionResult(np.full((36, 64), 255, np.uint8), 1.0, [(0, 0, 1, 1)])


def test_person_coming_into_view_is_confirmed_on_its_first_confident_frame():
    """Faint at first (dark doorway), then clear: every motion frame is detected and the faint hits count."""
    scores = [0.3, 0.35, 0.4, 0.6, 0.8]
    plan = [[Detection("person", s, (0.40 + i * 0.01, 0.3, 0.48 + i * 0.01, 0.6))] for i, s in enumerate(scores)]
    p = CameraPipeline(CAM, Scripted(plan))
    p.motion = Moving()
    results = [p.process(frame(i / 5)) for i in range(len(scores))]
    assert all(r.detections is not None for r in results)               # no frame skipped
    assert [bool(r.confirmed) for r in results] == [False, False, False, True, False]


def test_faint_static_track_does_not_heat_map():
    sm = StaticMap(hot_threshold=1)
    p = CameraPipeline(CAM, Scripted([[Detection("person", 0.35, LAMP)] for _ in range(100)]), sm)
    p.cfg.detect_refresh = 0
    run(p, 100)
    run(p, 25, t0=30)
    assert not sm.is_hot(LAMP, 40)


def test_person_mask_drops_detections_with_feet_inside():
    cam = CAM.model_copy(update={"person_mask": [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]})
    plan = [[Detection("person", 0.8, (0.1 + i * 0.03, 0.3, 0.18 + i * 0.03, 0.8))] for i in range(15)]
    p = CameraPipeline(cam, Scripted(plan))
    p.cfg.detect_refresh = 0
    assert run(p, 15) == []
