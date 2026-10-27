"""Per-camera memory of places where 'objects' sit without ever moving (lamps, tyres, a coat on a hook).

When a track of an alert label ends without ever having moved, its location gets heat, and so does
an event marked as a false alarm (enough to make the spot hot at once). Detections
in hot places need a higher score and real travel before they can raise an alert. Heat decays,
so a place cools down again if nothing false shows up there.
"""

import time
from pathlib import Path

import numpy as np

GRID = (18, 32)  # rows, cols


class StaticMap:
    def __init__(self, path: Path | None = None, *, half_life_days: float = 3.0, hot_threshold: float = 3.0):
        self.path = path
        self.half_life = half_life_days * 86400
        self.hot_threshold = hot_threshold
        self.heat = np.zeros(GRID, np.float32)
        self.updated = time.time()
        if path and path.exists():
            data = np.load(path)
            self.heat, self.updated = data["heat"], float(data["updated"])

    def _cells(self, box: tuple) -> tuple[slice, slice]:
        rows, cols = GRID
        # The box's central half, so a big box doesn't heat its whole surroundings.
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        x1, x2 = cx - w / 4, cx + w / 4
        y1, y2 = cy - h / 4, cy + h / 4
        return (slice(int(y1 * rows), max(int(np.ceil(y2 * rows)), int(y1 * rows) + 1)),
                slice(int(x1 * cols), max(int(np.ceil(x2 * cols)), int(x1 * cols) + 1)))

    def _decay(self, now: float) -> None:
        if now > self.updated:
            self.heat *= 0.5 ** ((now - self.updated) / self.half_life)
            self.updated = now

    def add(self, box: tuple, now: float, weight: float = 1.0) -> None:
        self._decay(now)
        self.heat[self._cells(box)] += weight
        np.maximum(self.heat, 0, out=self.heat)

    def is_hot(self, box: tuple, now: float) -> bool:
        self._decay(now)
        cells = self.heat[self._cells(box)]
        return bool(cells.size and cells.max() >= self.hot_threshold)

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                np.savez(f, heat=self.heat, updated=self.updated)
