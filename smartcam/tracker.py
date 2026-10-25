"""Object tracking plus the per-track evidence that decides what is really moving.

Frigate attaches motion to whatever object overlaps it, so a person walking past a parked car (or a
lamp) can turn the car (or lamp) into the "moving" object. Here every track carries its own evidence:

- travel:     how far its box centre has wandered, relative to its own size. A parked car or a lamp
              has ~0 no matter what moves around it.
- stationary: a track that has not travelled for `stationary_after` seconds. It is never an alert
              subject and in-box motion no longer counts for it, until it really travels again.
- motion ratio: share of the box covered by the motion mask. Lets a person who stays in place
              (at a door, by a car) count as active, but only while the track is not stationary.
"""

import itertools
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .detect import Detection
from .motion import MotionResult

# Labels that may swap on the same object from frame to frame (a car seen as a truck).
GROUPS = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
          "cat": "animal", "dog": "animal", "person": "person"}


@dataclass
class TrackerConfig:
    new_track_score: float = 0.5     # a track must start from a confident detection (ByteTrack-style)
    match_score: float = 0.3         # weaker detections may only extend existing tracks
    max_age: float = 3.0             # seconds without a detection before a moving track ends
    stationary_max_age: float = 60.0 # stationary objects survive longer gaps (occlusion, detection cadence)
    stationary_after: float = 30.0
    stationary_travel: float = 0.15
    moving_travel: float = 0.25
    active_motion_ratio: float = 0.2
    active_motion_frames: int = 3


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    first_ts: float
    cfg: TrackerConfig
    history: deque = field(default_factory=lambda: deque(maxlen=600))   # (ts, box, score)
    motion: deque = field(default_factory=lambda: deque(maxlen=10))     # in-box motion ratio per detection
    labels: Counter = field(default_factory=Counter)
    stationary: bool = False
    last_ts: float = 0.0
    max_score: float = 0.0

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0]

    @property
    def box(self) -> tuple:
        return self.history[-1][1]

    @property
    def hits(self) -> int:
        return len(self.history)

    @property
    def age(self) -> float:
        return self.last_ts - self.first_ts

    def travel(self, since: float | None = None) -> float:
        """Spread of the box centre (max of x and y range) relative to the median box size."""
        pts = [(ts, b) for ts, b, _ in self.history if since is None or ts >= since]
        if len(pts) < 2:
            return 0.0
        b = np.array([b for _, b in pts])
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        w, h = np.median(b[:, 2] - b[:, 0]), np.median(b[:, 3] - b[:, 1])
        return float(max(np.ptp(cx) / max(w, 1e-3), np.ptp(cy) / max(h, 1e-3)))

    @property
    def moving(self) -> bool:
        """Actively doing something: travelling, or (if not stationary) moving in place."""
        if self.travel() >= self.cfg.moving_travel and not self.stationary:
            return True
        if self.stationary:
            return False
        recent = list(self.motion)[-self.cfg.active_motion_frames:]
        return len(recent) == self.cfg.active_motion_frames and min(recent) >= self.cfg.active_motion_ratio

    def add(self, ts: float, det: Detection, motion: MotionResult | None) -> None:
        self.history.append((ts, det.box, det.score))
        self.labels[det.label] += 1
        self.last_ts = ts
        self.max_score = max(self.max_score, det.score)
        self.motion.append(motion.ratio_in(det.box) if motion is not None else 0.0)
        self._update_stationary(ts)

    def _update_stationary(self, ts: float) -> None:
        cfg = self.cfg
        if self.stationary:
            # Wake up only on real travel in the last 10 s, not on motion around the object.
            if self.travel(since=ts - 10) >= cfg.moving_travel * 2:
                self.stationary = False
        elif self.age >= cfg.stationary_after and self.travel(since=ts - cfg.stationary_after) < cfg.stationary_travel:
            self.stationary = True


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, ts: float, dets: list[Detection], motion: MotionResult | None = None) -> tuple[list[Track], list[Track]]:
        """Associate detections with tracks. Returns (active tracks, tracks that just ended)."""
        dets = [d for d in dets if d.score >= self.cfg.match_score]
        pairs = []
        for t in self.tracks.values():
            for j, d in enumerate(dets):
                if GROUPS.get(d.label, d.label) != GROUPS.get(t.label, t.label):
                    continue
                score = self._affinity(t.box, d.box)
                if score > 0:
                    pairs.append((score, d.score, t.id, j))

        used_tracks, used_dets = set(), set()
        for _, _, tid, j in sorted(pairs, reverse=True):
            if tid in used_tracks or j in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(j)
            self.tracks[tid].add(ts, dets[j], motion)

        for j, d in enumerate(dets):
            if j not in used_dets and d.score >= self.cfg.new_track_score:
                t = Track(next(self._ids), ts, self.cfg)
                t.add(ts, d, motion)
                self.tracks[t.id] = t

        ended = []
        for tid, t in list(self.tracks.items()):
            max_age = self.cfg.stationary_max_age if t.stationary else self.cfg.max_age
            if ts - t.last_ts > max_age:
                ended.append(self.tracks.pop(tid))
        return list(self.tracks.values()), ended

    @staticmethod
    def _affinity(a: tuple, b: tuple) -> float:
        """IoU, falling back to centre distance so fast movers at 5 fps (no overlap) still match."""
        overlap = iou(a, b)
        if overlap > 0.1:
            return 1 + overlap
        diag = np.hypot(a[2] - a[0], a[3] - a[1])
        dist = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
        return max(0.0, 1 - dist / max(diag, 1e-3)) if dist < diag else 0.0
