import numpy as np

from smartcam.plates import PlateReader


class Ocr:
    def __init__(self, text, conf):
        self.text, self.confidence = text, conf


class Result:
    def __init__(self, text, conf):
        self.ocr = Ocr(text, conf)


def test_most_confident_plate_above_the_threshold():
    r = PlateReader(min_confidence=0.6)
    r._alpr = type("A", (), {"predict": lambda self, img: [Result("AB12345", [0.9, 0.95, 1.0]), Result("XX", [0.3, 0.4])]})()
    assert r.read(np.zeros((10, 10, 3), np.uint8)) == ("AB12345", 0.95)
    r._alpr = type("A", (), {"predict": lambda self, img: [Result("XX", [0.3, 0.4])]})()
    assert r.read(np.zeros((10, 10, 3), np.uint8)) is None
