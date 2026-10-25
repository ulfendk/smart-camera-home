"""Vehicle speed from two calibration lines across the road.

Per camera, two lines are drawn across the road a known distance apart (`speed.line_a`, `speed.line_b`,
`speed.distance_m`). A vehicle's ground point (bottom centre of its box) is followed from frame to frame;
the moment it crosses each line is interpolated between the two frames on either side, which matters at
5 fps (a car at 50 km/h covers 2.8 m between frames). Speed = distance / time between the crossings.

The road is assumed straight and level between the lines, and the lines should be drawn where the
vehicle's wheels touch the road. Accuracy is limited by the box's bottom edge; a long base helps.
"""

from dataclasses import dataclass

from pydantic import BaseModel, field_validator

VEHICLES = {"car", "truck", "bus", "motorcycle"}


class SpeedConfig(BaseModel):
    line_a: list[float] | None = None      # x1, y1, x2, y2 (normalised)
    line_b: list[float] | None = None
    distance_m: float = 0                  # between the lines, along the road
    limit_kmh: float | None = None         # shown with the reading
    notify_over_kmh: float | None = None   # push notification at or above this speed
    read_plates: bool = True               # read the licence plate on the best shot

    @field_validator("line_a", "line_b")
    @classmethod
    def _line(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("a line is 4 numbers: x1, y1, x2, y2")
        return v

    @property
    def ready(self) -> bool:
        return bool(self.line_a and self.line_b and self.distance_m > 0)


@dataclass
class SpeedReading:
    camera: str
    track_id: int
    label: str
    kmh: float
    t_a: float          # when it crossed line A
    t_b: float
    box: tuple          # its box nearest the middle of the measurement
    score: float
    direction: str      # "a→b" or "b→a"


def _ground(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, box[3])


def crossing(p, q, line) -> float | None:
    """Where segment p→q crosses the line segment, as a fraction of p→q (None if it does not)."""
    (x1, y1), (x2, y2) = p, q
    x3, y3, x4, y4 = line
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    s = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    return s if 0 <= s <= 1 and 0 <= u <= 1 else None


class SpeedMeter:
    def __init__(self, camera: str, sc: SpeedConfig):
        self.camera, self.sc = camera, sc
        self.crossed: dict[int, dict[str, float]] = {}
        self.last: dict[int, tuple] = {}     # track id -> last history entry handled
        self.done: set[int] = set()

    def update(self, tracks) -> list[SpeedReading]:
        out = []
        live = set()
        for t in tracks:
            live.add(t.id)
            if t.label not in VEHICLES or t.id in self.done or len(t.history) < 2:
                continue
            marks = self.crossed.setdefault(t.id, {})
            prev = self.last.get(t.id)
            for h in t.history:
                if prev is not None and h[0] <= prev[0]:
                    continue
                if prev is not None:
                    (t0, b0, _), (t1, b1, _) = prev, h
                    for name, line in (("a", self.sc.line_a), ("b", self.sc.line_b)):
                        if name not in marks and (s := crossing(_ground(b0), _ground(b1), line)) is not None:
                            marks[name] = t0 + s * (t1 - t0)
                prev = h
            self.last[t.id] = prev
            if "a" in marks and "b" in marks:
                dt = abs(marks["b"] - marks["a"])
                self.done.add(t.id)
                if 0.05 <= dt <= 30:
                    mid = (marks["a"] + marks["b"]) / 2
                    box = min(t.history, key=lambda h: abs(h[0] - mid))[1]
                    out.append(SpeedReading(self.camera, t.id, t.label, round(self.sc.distance_m / dt * 3.6, 1),
                                            marks["a"], marks["b"], box, t.max_score,
                                            "a→b" if marks["a"] < marks["b"] else "b→a"))
        for gone in [i for i in self.crossed if i not in live]:
            self.crossed.pop(gone, None)
            self.last.pop(gone, None)
            self.done.discard(gone)
        return out
