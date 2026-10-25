"""Motion detection that ignores lighting changes (lamps, sun, clouds, IR/colour switches).

Two defences:
- Illumination-invariant comparison: the frame is gain-compensated to the background's median, and
  a pixel only counts as motion if both its brightness and its brightness *relative to its
  neighbourhood* changed. Light from a lamp or the sun is smooth, so it raises a pixel and its
  neighbourhood together and the ratio stays put; an object entering changes it.
- Lighting gate: if a large part of the frame changes at once, or the global brightness jumps,
  it is treated as a lighting event: the background is reset and no motion is reported.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_WIDTH = 320


@dataclass
class MotionResult:
    mask: np.ndarray                 # uint8 0/255 at work resolution
    fraction: float                  # share of (unmasked) pixels in motion
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # normalised x1,y1,x2,y2
    lighting: bool = False           # frame was a lighting event, motion suppressed

    @property
    def active(self) -> bool:
        return bool(self.boxes)

    def ratio_in(self, box: tuple[float, float, float, float]) -> float:
        """Share of pixels inside a normalised box that are in motion."""
        h, w = self.mask.shape
        x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(np.ceil(box[2] * w)), int(np.ceil(box[3] * h))
        roi = self.mask[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
        return float(np.count_nonzero(roi)) / roi.size if roi.size else 0.0


def polygon_mask(shape: tuple[int, int], polygons: list[list[tuple[float, float]]]) -> np.ndarray:
    """255 inside any of the normalised polygons, 0 elsewhere."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for poly in polygons:
        pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


class MotionDetector:
    def __init__(self, ignore: list[tuple[float, float]] | None = None, *,
                 diff_threshold: int = 25, ratio_threshold: float = 0.15, min_area: float = 0.0008,
                 lighting_fraction: float = 0.35, lighting_jump: float = 0.12, alpha: float = 0.05):
        self.ignore_poly = ignore
        self.diff_threshold, self.ratio_threshold = diff_threshold, ratio_threshold
        self.min_area, self.lighting_fraction, self.lighting_jump, self.alpha = (
            min_area, lighting_fraction, lighting_jump, alpha)
        self.bg: np.ndarray | None = None
        self.bg_ratio: np.ndarray | None = None
        self.keep: np.ndarray | None = None  # 1 where motion counts

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        size = (WORK_WIDTH, round(h * WORK_WIDTH / w))
        gray = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)

    @staticmethod
    def _ratio(gray: np.ndarray) -> np.ndarray:
        """Brightness relative to the local neighbourhood (~1/20 of the frame width)."""
        return gray / (cv2.GaussianBlur(gray, (0, 0), 8) + 8)

    def _reset(self, gray: np.ndarray) -> None:
        self.bg, self.bg_ratio = gray.copy(), self._ratio(gray)

    def update(self, image: np.ndarray) -> MotionResult:
        gray = self._prepare(image)
        if self.keep is None:
            self.keep = (polygon_mask(gray.shape, [self.ignore_poly]) == 0) if self.ignore_poly else np.ones(gray.shape, bool)
        if self.bg is None:
            self._reset(gray)
            return MotionResult(np.zeros(gray.shape, np.uint8), 0.0)

        bg_median = float(np.median(self.bg)) + 1
        gain = bg_median / (float(np.median(gray)) + 1)
        jump = abs(gain - 1)
        comp = gray * gain
        ratio = self._ratio(comp)

        changed = (np.abs(comp - self.bg) > self.diff_threshold) & (np.abs(ratio - self.bg_ratio) > self.ratio_threshold)
        changed &= self.keep
        mask = changed.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        fraction = float(np.count_nonzero(mask)) / max(int(self.keep.sum()), 1)

        if fraction > self.lighting_fraction or jump > self.lighting_jump:
            self._reset(gray)
            return MotionResult(np.zeros_like(mask), fraction, lighting=True)

        # Learn slowly, and not at all where something is moving, so a person standing still
        # is not absorbed into the background within a few frames.
        learn = np.where(mask > 0, 0.0, self.alpha).astype(np.float32)
        self.bg += (comp - self.bg) * learn
        self.bg_ratio += (ratio - self.bg_ratio) * learn

        h, w = mask.shape
        min_px = self.min_area * h * w
        boxes = []
        # The ratio test is weakest inside large uniform objects; fill each blob so its interior
        # counts too (needed for per-box motion ratios), and drop blobs too small to matter.
        filled = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= min_px:
                cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((x / w, y / h, (x + bw) / w, (y + bh) / h))
        return MotionResult(filled & (self.keep.astype(np.uint8) * 255), fraction, boxes)
