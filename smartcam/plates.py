"""Licence plates on vehicle close-ups (fast-alpr: a YOLOv9 plate detector and a plate OCR model,
both ONNX). Runs on the CPU so it never competes with the shared GPU detector; one close-up takes
a few tens of milliseconds. The models are downloaded once into ~/.cache (the Docker image ships them).
"""

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

DETECTOR = "yolo-v9-t-384-license-plate-end2end"
OCR = "cct-s-v2-global-model"


class PlateReader:
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self._alpr = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._alpr is None:
                from fast_alpr import ALPR
                self._alpr = ALPR(detector_model=DETECTOR, detector_providers=["CPUExecutionProvider"],
                                  ocr_model=OCR, ocr_device="cpu")
        return self._alpr

    def read(self, image: np.ndarray) -> tuple[str, float] | None:
        """The most confident plate in the image as (text, confidence 0-1), or None."""
        alpr = self.load()
        with self._lock:   # one ONNX session pair, called from worker threads
            results = alpr.predict(image)
        best = None
        for r in results:
            if not r.ocr or not r.ocr.text:
                continue
            conf = float(np.mean(r.ocr.confidence)) if r.ocr.confidence is not None else 0.0
            if conf >= self.min_confidence and (best is None or conf > best[1]):
                best = (r.ocr.text, round(conf, 2))
        return best


_reader: PlateReader | None = None


def reader() -> PlateReader:
    global _reader
    if _reader is None:
        _reader = PlateReader()
    return _reader
