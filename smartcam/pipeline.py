"""Per-camera processing: motion -> detection -> tracking -> confirmed alert objects."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CameraConfig
from .detect import Detection
from .ingest import Frame
from .motion import MotionDetector, MotionResult
from .static_map import StaticMap
from .tracker import Track, Tracker, TrackerConfig


@dataclass
class PipelineConfig:
    alert_labels: tuple[str, ...] = ("person",)
    confirm_hits: int = 3
    confirm_score: float = 0.5
    hot_confirm_score: float = 0.7   # in places the static map knows for false positives
    detect_refresh: float = 5.0      # detect at least this often so parked objects stay known
    idle_detect_interval: float = 0.4  # ongoing motion with nothing tracked (trees in wind): detect at most this often
    quiet_scene: float = 0.3         # ...but every frame where motion is rare (under this share of frames over
    busy_window: float = 60.0        # about this many seconds), so a person coming into view is caught at once
    static_min_age: float = 10.0     # a never-moving track must last this long to heat the static map


@dataclass
class FrameResult:
    frame: Frame
    motion: MotionResult
    detections: list[Detection] | None    # None when detection was skipped for this frame
    tracks: list[Track]
    confirmed: list[Track] = field(default_factory=list)  # tracks confirmed on this frame
    ended: list[Track] = field(default_factory=list)


def _inside(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    return cv2.pointPolygonTest(np.array(poly, np.float32), point, False) >= 0


class CameraPipeline:
    def __init__(self, cam: CameraConfig, detector, static_map: StaticMap | None = None,
                 cfg: PipelineConfig | None = None, tracker_cfg: TrackerConfig | None = None):
        self.cam, self.detector = cam, detector
        self.static_map = static_map or StaticMap()
        self.cfg = cfg or PipelineConfig()
        self.motion = MotionDetector(cam.motion_mask)
        self.tracker = Tracker(tracker_cfg)
        self.confirmed_ids: set[int] = set()
        self.ever_moved: set[int] = set()
        self.last_detect = float("-inf")
        self.busy, self.last_ts = 0.0, None   # share of recent frames with motion (moving average)

    def _filter(self, dets: list[Detection]) -> list[Detection]:
        if not self.cam.person_mask:
            return dets
        # Frigate semantics: a person whose feet are inside the mask is ignored.
        return [d for d in dets if not (d.label == "person" and _inside(d.bottom_center, self.cam.person_mask))]

    def process(self, frame: Frame) -> FrameResult:
        motion = self.motion.update(frame.image)
        if self.last_ts is not None:
            a = min(1.0, max(0.0, frame.ts - self.last_ts) / self.cfg.busy_window)
            self.busy += a * (motion.active - self.busy)
        self.last_ts = frame.ts
        since = frame.ts - self.last_detect
        tracking = any(not t.stationary for t in self.tracker.tracks.values())
        if tracking:
            due = True                                      # follow moving objects at full rate
        elif motion.active:
            quiet = self.busy < self.cfg.quiet_scene
            due = quiet or since >= self.cfg.idle_detect_interval   # motion, nothing tracked yet
        else:
            due = since >= self.cfg.detect_refresh
        if not due:
            return FrameResult(frame, motion, None, list(self.tracker.tracks.values()))

        self.last_detect = frame.ts
        dets = self._filter(self.detector.detect(frame.image))
        tracks, ended = self.tracker.update(frame.ts, dets, motion)

        confirmed = []
        for t in tracks:
            if t.moving:
                self.ever_moved.add(t.id)
            if t.id not in self.confirmed_ids and self._confirm(t, frame.ts):
                self.confirmed_ids.add(t.id)
                confirmed.append(t)
        for t in ended:
            # Only confident ones: faint tracks (started from 0.3) must not mark spots as false alarms.
            if (t.label in self.cfg.alert_labels and t.id not in self.ever_moved
                    and t.age >= self.cfg.static_min_age and t.max_score >= self.cfg.confirm_score):
                self.static_map.add(t.box, frame.ts)
            self.confirmed_ids.discard(t.id)
            self.ever_moved.discard(t.id)
        return FrameResult(frame, motion, dets, tracks, confirmed, ended)

    def _confirm(self, t: Track, now: float) -> bool:
        cfg = self.cfg
        if t.label not in cfg.alert_labels or t.hits < cfg.confirm_hits or not t.moving:
            return False
        if self.static_map.is_hot(t.box, now):
            # Known false-positive spot: demand a confident detection that actually travels.
            return t.max_score >= cfg.hot_confirm_score and t.travel() >= t.cfg.moving_travel
        return t.max_score >= cfg.confirm_score
