import cv2
import numpy as np

from smartcam.motion import MotionDetector


# Fixed textures so consecutive frames differ only by what the test changes. Real walls, floors and
# clothes are textured; the detector relies on that to tell objects from lighting.
_BG = cv2.GaussianBlur(np.random.default_rng(0).integers(30, 170, (360, 640, 3), dtype=np.uint8), (3, 3), 0)
_PERSON = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 90, (180, 40, 3), dtype=np.uint8), (3, 3), 0)


def scene(brightness: float = 1.0, person_x: int | None = None) -> np.ndarray:
    img = _BG.copy()
    cv2.rectangle(img, (100, 100), (220, 300), (200, 200, 200), -1)  # a cupboard
    if person_x is not None:
        img[150:330, person_x:person_x + 40] = _PERSON
    return np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


def warm(md: MotionDetector, n: int = 5) -> None:
    for _ in range(n):
        md.update(scene())


def test_static_scene_has_no_motion():
    md = MotionDetector()
    warm(md)
    assert not md.update(scene()).active


def test_brightness_ramp_is_not_motion():
    # A lamp or the sun slowly brightening the room: +2% per frame up to +40%.
    md = MotionDetector()
    warm(md)
    for i in range(1, 21):
        assert not md.update(scene(1 + 0.02 * i)).active


def test_sudden_lamp_switch_is_a_lighting_event():
    md = MotionDetector()
    warm(md)
    r = md.update(scene(1.5))
    assert r.lighting and not r.active
    assert not md.update(scene(1.5)).active  # background was reset to the new lighting


def test_walking_person_is_motion_and_located():
    md = MotionDetector()
    warm(md)
    r = None
    for x in range(300, 420, 20):
        r = md.update(scene(person_x=x))
    assert r.active and not r.lighting
    person_box = (400 / 640, 150 / 360, 440 / 640, 330 / 360)
    assert r.ratio_in(person_box) > 0.3
    assert r.ratio_in((100 / 640, 100 / 360, 220 / 640, 300 / 360)) < 0.05  # cupboard untouched


def test_ignore_polygon_suppresses_motion():
    md = MotionDetector(ignore=[(0.4, 0.0), (1.0, 0.0), (1.0, 1.0), (0.4, 1.0)])
    warm(md)
    for x in range(300, 420, 20):
        r = md.update(scene(person_x=x))
    assert not r.active


def glow(img: np.ndarray, strength: float, cx: int = 450, cy: int = 120, sigma: float = 70) -> np.ndarray:
    """A lamp or sunbeam brightening one part of the room smoothly."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    light = 1 + strength * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)


def test_local_lamp_glow_is_not_motion():
    md = MotionDetector()
    warm(md)
    for strength in (0.2, 0.4, 0.6, 0.8):  # lamp warming up / sun patch strengthening over a few frames
        r = md.update(glow(scene(), strength))
        assert not r.active, f"glow {strength} reported as motion: {r.boxes}"


def test_person_still_detected_inside_lamp_glow():
    md = MotionDetector()
    for _ in range(5):
        md.update(glow(scene(), 0.6))
    for x in range(380, 460, 20):
        r = md.update(glow(scene(person_x=x), 0.6))
    assert r.active
