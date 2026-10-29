"""Who is it: faces, and the same person carried from camera to camera.

While a person event is open, SmartCam takes a few full-resolution stills of the person (go2rtc, main stream).
- On cameras with `recognize: true` it looks for a face in them: YuNet (MIT, OpenCV Zoo) finds it and its five
  landmarks, the face is aligned to the ArcFace template, and AuraFace (Apache-2.0, an ArcFace ResNet100) turns it
  into a 512-number embedding, compared with the faces learned for each person (people.py).
- On every camera it keeps the person's colours (upper and lower body, hue and saturation only, so a darker camera
  matters less). A person named on one camera hands the name to an unnamed event on another camera within the
  sequence window if the colours match and nobody else fits: "probably Farmor (seen at the front door)". A face
  recognised later names the earlier events the same way.

Recognition only adds a name; it never makes anything less of an alarm.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel

from . import stills
from .config import Config
from .detect.yolo import providers
from .people import People

log = logging.getLogger(__name__)

# Where ArcFace-family models expect the eyes, nose tip and mouth corners in a 112 x 112 face.
ARCFACE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


class FacesConfig(BaseModel):
    enabled: bool = False
    model: Path = Path("models/auraface.onnx")
    detector: Path = Path("models/yunet.onnx")
    threshold: float = 0.45       # cosine similarity for "this is that person"
    min_face: int = 40            # px (face height in the still); smaller faces are too blurry to trust
    attempts: int = 6             # stills per event on a recognising camera (others: 3, for the colours only)
    interval: float = 1.0         # seconds between stills
    wait: float = 4.0             # WFH / armed home: hold a push this long for the name (family is not pushed)
    carry_distance: float = 0.35  # colour difference (Bhattacharyya, 0 same .. 1 different) for "the same person"


def align(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE, method=cv2.LMEDS)
    return cv2.warpAffine(image, m, (112, 112), borderValue=0)


def match(vec: np.ndarray, gallery: dict[str, np.ndarray], threshold: float) -> tuple[str | None, float]:
    """The best person for this embedding (its best stored face), or None when nobody reaches the threshold."""
    best, score = None, 0.0
    for name, faces in gallery.items():
        s = float((faces @ vec).max())
        if s > score:
            best, score = name, s
    return (best, score) if score >= threshold else (None, score)


def signature(person: np.ndarray) -> np.ndarray | None:
    """Colours of the upper and lower body (hue x saturation histograms), the middle of the box only."""
    h, w = person.shape[:2]
    if h < 40 or w < 10:
        return None
    hsv = cv2.cvtColor(person[:, int(w * .2):int(w * .8)], cv2.COLOR_BGR2HSV)
    out = []
    for top, bottom in ((.15, .5), (.5, .95)):
        hist = cv2.calcHist([hsv[int(h * top):int(h * bottom)]], [0, 1], None, [16, 8], [0, 180, 0, 256])
        out.append(cv2.normalize(hist, None, 1, 0, cv2.NORM_L1).flatten())
    return np.stack(out).astype(np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(cv2.compareHist(a[i], b[i], cv2.HISTCMP_BHATTACHARYYA) for i in range(2))


def crop(image: np.ndarray, box, grow: float = 0.0) -> np.ndarray:
    """A normalised box (x1, y1, x2, y2) of the image, grown by `grow` of its size on every side."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * grow, (y2 - y1) * grow
    return image[max(0, int((y1 - dy) * h)):min(h, int((y2 + dy) * h)), max(0, int((x1 - dx) * w)):min(w, int((x2 + dx) * w))]


class FaceRecognizer:
    def __init__(self, fc: FacesConfig, device: str = "GPU"):
        self.min_face = fc.min_face
        self.detector = cv2.FaceDetectorYN.create(str(fc.detector), "", (320, 320), 0.6, 0.3, 50)
        names, options = providers(ort.get_available_providers(), device)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(fc.model), sess_options=opts, providers=names, provider_options=options)
        self.input = self.session.get_inputs()[0].name
        self.lock = threading.Lock()   # one detector and session, used from worker threads
        used = self.session.get_providers()[0]
        log.info("%s on %s%s", fc.model.name, used, f" ({device})" if used == "OpenVINOExecutionProvider" else "")

    def best_face(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(embedding, face picture) of the largest face big enough to trust, or None."""
        h, w = image.shape[:2]
        if h < self.min_face or w < self.min_face:
            return None
        with self.lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
        if faces is None or not len(faces):
            return None
        f = max(faces, key=lambda f: f[2] * f[3])
        if f[3] < self.min_face:
            return None
        face = align(image, f[4:14].reshape(5, 2))
        blob = ((face[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis]
        with self.lock:
            vec = self.session.run(None, {self.input: blob})[0][0]
        x, y, fw, fh = (int(v) for v in f[:4])
        picture = image[max(0, y - fh // 4):y + fh + fh // 4, max(0, x - fw // 4):x + fw + fw // 4]
        return vec / np.linalg.norm(vec), picture


@dataclass
class Seen:
    """One person event as recognition sees it."""
    ev: object                    # events.OpenEvent
    camera: str
    start: float
    last: float
    attempts: int = 0
    next_at: float = 0.0
    busy: bool = False
    sig: np.ndarray | None = None
    sig_area: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Recognition:
    def __init__(self, cfg: Config, engine, people: People, fc: FacesConfig | None = None, model=None, grab=None):
        self.cfg, self.engine, self.people = cfg, engine, people
        self.fc = fc or FacesConfig(**cfg.faces)
        self.model = model if model is not None else FaceRecognizer(self.fc, cfg.detect.openvino_device)
        self.grab = grab or stills.grab
        self.http = httpx.AsyncClient(timeout=10)
        self.window = engine.rules.sequence_window
        self.seen: dict[int, Seen] = {}
        self._tasks: set[asyncio.Task] = set()
        (cfg.paths.data / "faces").mkdir(parents=True, exist_ok=True)
        engine.recognizer = self

    def recognizes(self, camera: str) -> bool:
        return self.cfg.cameras[camera].recognize

    def _max(self, s: Seen) -> int:
        return self.fc.attempts if self.recognizes(s.camera) else 3

    def see(self, ev, box, now: float) -> None:
        """The event's person was just seen at `box`; take a still now and then."""
        for k in [k for k, s in self.seen.items() if s.last < now - self.window * 5]:
            del self.seen[k]
        s = self.seen.get(ev.id)
        if s is None:
            s = self.seen[ev.id] = Seen(ev, ev.camera, ev.start, now)
        s.last = now
        if s.busy or s.attempts >= self._max(s) or now < s.next_at:
            return
        s.busy, s.attempts, s.next_at = True, s.attempts + 1, now + self.fc.interval
        task = asyncio.create_task(self._look(s, tuple(box)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait(self, ev_id: int, timeout: float) -> None:
        """Until the event is named, or has had all its stills, or `timeout`."""
        if (s := self.seen.get(ev_id)) is None:
            return
        try:
            await asyncio.wait_for(s.done.wait(), timeout)
        except TimeoutError:
            pass

    async def _look(self, s: Seen, box) -> None:
        try:
            image = await self.grab(self.cfg, self.http, s.camera)
            if image is None:
                return
            person = crop(image, box)
            if (sig := signature(person)) is not None and person.size >= s.sig_area:
                s.sig, s.sig_area = sig, person.size   # the biggest view of them is the best
            if self.recognizes(s.camera):
                found = await asyncio.to_thread(self.model.best_face, crop(image, box, 0.3))
                if found is not None:
                    await self._face(s, *found)
            if s.ev.person is None and s.sig is not None:
                await self._carry(s)
        except Exception:
            log.exception("recognition failed for event %d", s.ev.id)
        finally:
            s.busy = False
            if s.attempts >= self._max(s) or (s.ev.person and not s.ev.person_via):
                s.done.set()

    async def _face(self, s: Seen, vec: np.ndarray, picture: np.ndarray) -> None:
        rel = f"faces/{s.ev.id}_{s.attempts}.jpg"
        cv2.imwrite(str(self.cfg.paths.data / rel), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.people.add_face(s.ev.id, vec, rel, time.time())
        if s.attempts == 1 or not self.engine.db.event(s.ev.id).get("face"):
            self.engine.db.update_event(s.ev.id, face=rel)
        name, score = match(vec, self.people.gallery(), self.fc.threshold)
        if name and (s.ev.person_via or score > s.ev.person_score):   # a face beats a carried name
            await self.engine.set_person(s.ev, name, self.people.group(name), score, None)
            await self._hand_back(s)

    def _near(self, a: Seen, b: Seen) -> bool:
        return a.start <= b.last + self.window and b.start <= a.last + self.window

    def _fits(self, s: Seen) -> set[str]:
        """Names of people seen on other cameras around the same time who look like this event's person."""
        return {o.ev.person for o in self.seen.values()
                if o is not s and o.camera != s.camera and o.ev.person and o.sig is not None
                and self._near(o, s) and distance(o.sig, s.sig) <= self.fc.carry_distance}

    async def _carry(self, s: Seen) -> None:
        names = self._fits(s)
        if len(names) != 1:
            return   # nobody, or two people who both fit: no guessing
        name = names.pop()
        source = max((o for o in self.seen.values() if o.ev.person == name and o.camera != s.camera and o.sig is not None
                      and self._near(o, s)), key=lambda o: o.last)
        await self.engine.set_person(s.ev, name, self.people.group(name), 0.0, source.camera)
        s.done.set()

    async def _hand_back(self, s: Seen) -> None:
        """A face named this event: unnamed events on other cameras around the same time that look alike get the name."""
        if s.sig is None:
            return
        for o in list(self.seen.values()):
            if o is not s and o.camera != s.camera and o.ev.person is None and o.sig is not None and self._near(o, s) \
                    and distance(o.sig, s.sig) <= self.fc.carry_distance and self._fits(o) == {s.ev.person}:
                await self.engine.set_person(o.ev, s.ev.person, s.ev.person_group, 0.0, s.camera)
                o.done.set()
