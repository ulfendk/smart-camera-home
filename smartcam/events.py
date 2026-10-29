"""Turn confirmed tracks into events, score how suspicious they are, and hand them to notifiers.

An event opens when a track is confirmed (see pipeline.py) and closes when the track ends. While it
is open its suspicion score is updated from rules; once the score reaches the notify threshold the
notifiers get an alert straight away with a snapshot, and the clip follows after the event ends.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from .clips import cut_clip, frame_at, frames_between
from .speed import VEHICLES, SpeedConfig
from .config import Config
from .db import Database
from .pipeline import FrameResult
from .static_map import StaticMap
from .tracker import Track, iou

log = logging.getLogger(__name__)


class RulesConfig(BaseModel):
    notify_score: float = 4
    loiter_seconds: float = 30
    door_zones: list[str] = []
    night_start: int = 22          # local hour
    night_end: int = 6
    sequence_window: float = 60    # person on another camera this recently adds suspicion
    pre_capture: float = 5
    post_capture: float = 5
    merge_gap: float = 30          # a person within this long after an event ends continues that event
    home_armed_bonus: float = 1    # armed at home: outdoor people score this much more


@dataclass
class OpenEvent:
    id: int
    camera: str
    location: str
    start: float
    last_ts: float
    zone_dwell: dict[str, float] = field(default_factory=dict)
    best_score: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    notified: bool = False
    tracks: set[int] = field(default_factory=set)
    closed_at: float | None = None     # last track ended; final once merge_gap passes without a new one
    person: str | None = None          # who it is (faces.py), if known
    person_group: str | None = None    # family, friend, neighbour, other
    person_score: float = 0.0          # face similarity; 0 for a name carried from another camera
    person_via: str | None = None      # the camera the name was carried from


def is_night(ts: float, rules: RulesConfig) -> bool:
    hour = datetime.fromtimestamp(ts).hour
    if rules.night_start > rules.night_end:
        return hour >= rules.night_start or hour < rules.night_end
    return rules.night_start <= hour < rules.night_end


def score_event(ev: OpenEvent, now: float, away: bool, rules: RulesConfig,
                recent_other: list[str], home_armed: bool = False, wfh: bool = False) -> tuple[float, list[str]]:
    score, reasons = 1.0, ["person"]
    if wfh and not away and ev.location == "outdoor":
        reasons.append("working from home")
    if away:
        if ev.location == "indoor":
            return 10.0, ["person indoors while away"]
        score += 3
        reasons.append("away")
    elif home_armed and ev.location == "outdoor":
        score += rules.home_armed_bonus
        reasons.append("armed at home")
    if is_night(now, rules):
        score += 1
        reasons.append("night")
    if ev.zone_dwell:
        zone, dwell = max(ev.zone_dwell.items(), key=lambda kv: kv[1])
        if dwell >= rules.loiter_seconds:
            score += 2
            reasons.append(f"loitering in {zone} ({dwell:.0f}s)")
    if any(z in ev.zone_dwell for z in rules.door_zones):
        score += 1
        reasons.append("at door")
    if recent_other:
        score += 1
        reasons.append(f"seen on {', '.join(recent_other)} just before")
    # Who it is changes nothing in the score: a known person in the house while away is still an alarm.
    if ev.person and ev.person_via:
        reasons.append(f"probably {ev.person} (seen at {ev.person_via})")
    elif ev.person:
        reasons.append(f"known: {ev.person} ({ev.person_group or 'other'})")
    return score, reasons


def should_notify(ev: OpenEvent, away: bool, now: float, rules: RulesConfig, home_armed: bool = False,
                  wfh: bool = False) -> bool:
    if wfh and ev.location == "outdoor":
        return True    # working from home: every person outside on the property
    if ev.score < rules.notify_score:
        return False
    # At home indoor events are family. Outdoors: at night, or at any time when armed at home.
    return away or (ev.location == "outdoor" and (home_armed or is_night(now, rules)))


def _in_zone(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    return cv2.pointPolygonTest(np.array(poly, np.float32), point, False) >= 0


Notifier = Callable[[dict], Awaitable[None]]


class EventEngine:
    def __init__(self, cfg: Config, db: Database, rules: RulesConfig | None = None):
        self.cfg, self.db = cfg, db
        self.rules = rules or RulesConfig()
        self.open: dict[tuple[str, int], OpenEvent] = {}   # (camera, track id) -> its event
        self.latest: dict[str, OpenEvent] = {}             # camera -> most recent event, for merging
        self._tasks: set[asyncio.Task] = set()             # keep references so pending tasks aren't GC'd
        self.recent_starts: list[tuple[float, str]] = []   # (ts, camera) of recent event starts
        self.on_open: list[Notifier] = []
        self.on_close: list[Notifier] = []
        self.on_alert: list[Notifier] = []
        self.on_clip: list[Notifier] = []
        self.on_away: list[Callable[[bool], Awaitable[None]]] = []
        self.on_wfh: list[Callable[[bool], Awaitable[None]]] = []
        self.on_score: list[Callable[[OpenEvent], Awaitable[None]]] = []   # when an event's score rises
        self.on_speed: list[Notifier] = []   # a vehicle speed event, once its best shot is in
        self.detector = None                 # set by detection; finds the vehicle in the recording
        self.people_seen: dict[str, float] = {}   # camera -> last time a confirmed person was in view (live view)
        self.people_boxes: dict[str, tuple] = {}  # camera -> (ts, boxes of people in view), for the wall feed
        self.static_maps: dict[str, StaticMap] = {}   # camera -> its static map, set by detection
        self.live_frames: dict[str, tuple] = {}   # camera -> (ts, newest image), for the wall feed
        self.on_occupancy: list[Callable[[str, bool], Awaitable[None]]] = []   # (camera, occupied)
        self.on_person: list[Notifier] = []    # an event's person was named (faces.py)
        self.recognizer = None                 # faces.Recognition, when recognition is on
        self.snapshots = cfg.paths.data / "snapshots"
        self.clips = cfg.paths.data / "clips"

    @property
    def away(self) -> bool:
        return self.db.get_setting("away", "0") == "1"

    @property
    def home_armed(self) -> bool:
        return self.db.get_setting("home_armed", "0") == "1"

    async def set_home_armed(self, armed: bool) -> None:
        self.db.set_setting("home_armed", "1" if armed else "0")
        if armed:
            self.db.set_setting("away_since", str(time.time()))   # only new activity counts, as for away
        log.info("armed at home %s", "on" if armed else "off")

    @property
    def wfh(self) -> bool:
        return self.db.get_setting("wfh", "0") == "1"

    async def set_wfh(self, on: bool) -> None:
        self.db.set_setting("wfh", "1" if on else "0")
        if on:
            self.db.set_setting("away_since", str(time.time()))   # only people arriving from now on are pushed
        log.info("working from home %s", "on" if on else "off")
        for cb in self.on_wfh:
            try:
                await cb(on)
            except Exception:
                log.exception("wfh callback failed")

    @property
    def away_since(self) -> float:
        return float(self.db.get_setting("away_since", "0"))

    async def set_away(self, away: bool) -> None:
        self.db.set_setting("away", "1" if away else "0")
        if away:
            self.db.set_setting("away_since", str(time.time()))
        log.info("away mode %s", "on" if away else "off")
        if away and self.wfh:
            await self.set_wfh(False)    # nobody is home to work there
        for cb in self.on_away:
            try:
                await cb(away)
            except Exception:
                log.exception("away callback failed")

    async def handle(self, res: FrameResult) -> None:
        cam = self.cfg.cameras[res.frame.camera]
        now = res.frame.ts
        for t in res.confirmed:
            if (ev := self._continue(cam.name, t, now)) is None:
                ev = self._open(t, res, cam.location)
                event = self.db.event(ev.id)
                for cb in self.on_open:
                    await self._safe(cb, event)
        for t in res.tracks:
            ev = self.open.get((cam.name, t.id))
            if ev is None or t.last_ts != now:
                continue
            self._update_zones(ev, t, now)
            if self.recognizer is not None:
                self.recognizer.see(ev, t.box, now)
            if t.history[-1][2] > ev.best_score:
                ev.best_score = t.history[-1][2]
                self._snapshot(ev, res, t)
            await self._rescore(ev, now)
        for t in res.ended:
            if (ev := self.open.pop((cam.name, t.id), None)) is not None:
                ev.tracks.discard(t.id)
                if not ev.tracks:
                    ev.closed_at = t.last_ts
                    self.db.update_event(ev.id, end_ts=t.last_ts)
                    task = asyncio.create_task(self._finalise(ev, t.last_ts))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

    def learn_review(self, event: dict, review: str | None) -> None:
        """A false alarm makes its spot hot in the camera's static map, so the same thing there needs a
        confident detection that really travels next time. Taking the mark back cools the spot again."""
        sm = self.static_maps.get(event["camera"])
        was, now = event["review"] == "false_positive", review == "false_positive"
        if sm is None or not event.get("box") or event["label"] != "person" or was == now:
            return
        sm.add(tuple(event["box"]), time.time(), sm.hot_threshold if now else -sm.hot_threshold)
        sm.save()

    def _continue(self, camera: str, t: Track, now: float) -> OpenEvent | None:
        """Attach a newly confirmed track to the camera's ongoing or just-ended event, if any.

        People sitting still or stepping out of view briefly lose their track; without this every
        reappearance would be a new event (and, while away, a new notification).
        """
        ev = self.latest.get(camera)
        if ev is None or (ev.closed_at is not None and now - ev.closed_at > self.rules.merge_gap):
            return None
        if ev.closed_at is not None:
            ev.closed_at = None
            self.db.update_event(ev.id, end_ts=None)
        ev.tracks.add(t.id)
        self.open[(camera, t.id)] = ev
        return ev

    def _open(self, t: Track, res: FrameResult, location: str) -> OpenEvent:
        now = res.frame.ts
        start = t.first_ts
        ev_id = self.db.insert_event(camera=res.frame.camera, label=t.label, start_ts=start, max_score=t.max_score)
        ev = OpenEvent(ev_id, res.frame.camera, location, start, now, best_score=t.max_score, tracks={t.id})
        self.open[(res.frame.camera, t.id)] = ev
        self.latest[res.frame.camera] = ev
        self.recent_starts.append((start, res.frame.camera))
        self._snapshot(ev, res, t)
        log.info("event %d: %s on %s", ev_id, t.label, res.frame.camera)
        return ev

    def _update_zones(self, ev: OpenEvent, t: Track, now: float) -> None:
        dt = now - ev.last_ts
        ev.last_ts = now
        feet = ((t.box[0] + t.box[2]) / 2, t.box[3])
        for zone, poly in self.cfg.cameras[ev.camera].zones.items():
            if _in_zone(feet, poly):
                ev.zone_dwell[zone] = ev.zone_dwell.get(zone, 0.0) + dt

    async def _rescore(self, ev: OpenEvent, now: float) -> None:
        cutoff = ev.start - self.rules.sequence_window
        self.recent_starts = [(ts, c) for ts, c in self.recent_starts if ts >= now - 3600]
        others = sorted({c for ts, c in self.recent_starts if c != ev.camera and cutoff <= ts <= ev.start})
        before = ev.score
        home, wfh = self.home_armed, self.wfh
        ev.score, ev.reasons = score_event(ev, now, self.away, self.rules, others, home_armed=home, wfh=wfh)
        if ev.score > before:
            for cb in self.on_score:
                try:
                    await cb(ev)
                except Exception:
                    log.exception("score callback failed for event %d", ev.id)
        self.db.update_event(ev.id, score=ev.score, reasons=ev.reasons, zones=sorted(ev.zone_dwell),
                             max_score=ev.best_score)
        # While away, only activity that started after away mode began is news (not the people leaving).
        fresh = not (self.away or home or wfh) or ev.start >= self.away_since
        if not ev.notified and fresh and should_notify(ev, self.away, now, self.rules, home_armed=home, wfh=wfh):
            ev.notified = True
            if self.recognizer is not None and (home or wfh) and not self.away:
                self._spawn(self._push_unless_family(ev))   # family at home is not news
            else:
                await self._push(ev)

    async def _push(self, ev: OpenEvent) -> None:
        self.db.update_event(ev.id, notified=1)
        event = self.db.event(ev.id)
        for notify in self.on_alert:
            await self._safe(notify, event)

    async def _push_unless_family(self, ev: OpenEvent) -> None:
        await self.recognizer.wait(ev.id, self.recognizer.fc.wait)
        if ev.person_group == "family":
            log.info("event %d: %s, family: not pushed", ev.id, ev.person)
            return
        await self._push(ev)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def set_person(self, ev: OpenEvent, name: str, group: str | None, score: float, via: str | None) -> None:
        """Recognition named the event's person (via: carried from that camera)."""
        ev.person, ev.person_group, ev.person_score, ev.person_via = name, group, score, via
        self.db.update_event(ev.id, person=name, person_score=score, person_via=via)
        log.info("event %d: %s%s", ev.id, name, f" (carried from {via})" if via else f" ({score:.2f})")
        await self._rescore(ev, ev.last_ts)
        event = {**self.db.event(ev.id), "person_group": group}
        for cb in self.on_person:
            await self._safe(cb, event)

    def _snapshot(self, ev: OpenEvent, res: FrameResult, t: Track) -> None:
        img = res.frame.image.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(t.box, (w, h, w, h)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        path = self.snapshots / f"{ev.id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.db.update_event(ev.id, snapshot=str(path.relative_to(self.cfg.paths.data)), box=list(t.box))

    async def _finalise(self, ev: OpenEvent, closed_at: float) -> None:
        """Close the event for good unless a new track continued it within merge_gap."""
        await asyncio.sleep(self.rules.merge_gap)
        if ev.closed_at != closed_at:
            return  # continued by a later track; that track's end schedules a new finalise
        log.info("event %d ended after %.0fs, score %.0f (%s)", ev.id, closed_at - ev.start, ev.score,
                 ", ".join(ev.reasons))
        if self.latest.get(ev.camera) is ev:
            del self.latest[ev.camera]
        event = self.db.event(ev.id)
        for cb in self.on_close:
            await self._safe(cb, event)
        await self._make_clip(ev.id, ev.camera, ev.start, closed_at)

    async def _make_clip(self, ev_id: int, camera: str, start: float, end: float) -> None:
        clip_start, clip_end = start - self.rules.pre_capture, end + self.rules.post_capture
        # Wait for the recorder to finish the segment containing the clip's end.
        await asyncio.sleep(max(0.0, clip_end + self.cfg.record.segment_seconds + 3 - time.time()))
        out = self.clips / f"{ev_id}.mp4"
        try:
            path = await asyncio.to_thread(cut_clip, self.cfg, camera, clip_start, clip_end, out)
        except Exception:
            log.exception("clip for event %d failed", ev_id)
            return
        if path is None:
            log.warning("no recording covers event %d", ev_id)
            return
        self.db.update_event(ev_id, clip=str(Path(path).relative_to(self.cfg.paths.data)))
        event = self.db.event(ev_id)
        if event["notified"]:
            for notify in self.on_clip:
                await self._safe(notify, event)

    async def record_speed(self, r, frame) -> int:
        """A vehicle speed reading becomes an event: snapshot now, best shot and clip once recorded."""
        ev_id = self.db.insert_event(camera=r.camera, label=r.label, start_ts=r.t_a if r.t_a < r.t_b else r.t_b,
                                     max_score=r.score)
        self.db.update_event(ev_id, end_ts=max(r.t_a, r.t_b), speed=r.kmh,
                             reasons=[f"{r.kmh:.0f} km/h", r.direction])
        img = frame.image.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(r.box, (w, h, w, h)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{r.kmh:.0f} km/h", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        path = self.snapshots / f"{ev_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.db.update_event(ev_id, snapshot=str(path.relative_to(self.cfg.paths.data)))
        log.info("event %d: %s on %s at %.0f km/h (%s)", ev_id, r.label, r.camera, r.kmh, r.direction)
        task = asyncio.create_task(self._speed_media(ev_id, r))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return ev_id

    async def _speed_media(self, ev_id: int, r) -> None:
        """Best shot from the full-resolution recording, a close crop for the plate and driver, the clip."""
        mid = (r.t_a + r.t_b) / 2
        await asyncio.sleep(max(0.0, mid + self.cfg.record.segment_seconds + 3 - time.time()))
        full = self.snapshots / f"{ev_id}_full.jpg"
        try:
            found = await asyncio.to_thread(self._find_vehicle, r, mid)
            if found is None and await asyncio.to_thread(frame_at, self.cfg, r.camera, mid, full):
                found = cv2.imread(str(full)), r.box
            if found is not None:
                img, (x1, y1, x2, y2) = found
                h, w = img.shape[:2]
                mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                crop = img[max(int((y1 - my) * h), 0):min(int((y2 + my) * h), h), max(int((x1 - mx) * w), 0):min(int((x2 + mx) * w), w)]
                crop_path = self.snapshots / f"{ev_id}_vehicle.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), 3)
                cv2.putText(img, f"{r.kmh:.0f} km/h", (int(x1 * w), max(int(y1 * h) - 12, 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
                cv2.imwrite(str(full), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                self.db.update_event(ev_id, snapshot=str(full.relative_to(self.cfg.paths.data)),
                                     crop=str(crop_path.relative_to(self.cfg.paths.data)))
                if SpeedConfig(**(self.cfg.cameras[r.camera].speed or {})).read_plates and crop.size:
                    from .plates import reader
                    if plate := await asyncio.to_thread(reader().read, crop):
                        self.db.update_event(ev_id, plate=plate[0], plate_conf=plate[1],
                                             reasons=[f"{r.kmh:.0f} km/h", r.direction, plate[0]])
                        log.info("event %d: plate %s (%.0f%%)", ev_id, plate[0], plate[1] * 100)
        except Exception:
            log.exception("best shot for speed event %d failed", ev_id)
        event = self.db.event(ev_id)
        for cb in self.on_speed:
            await self._safe(cb, event)
        await self._make_clip(ev_id, r.camera, min(r.t_a, r.t_b) - 2, max(r.t_a, r.t_b) + 2)

    def _find_vehicle(self, r, mid: float, window: float = 2.5):
        """The recorded frame where the vehicle is where it was measured, and its box there.

        Recording and detection clocks differ by up to a second or two (segment names are whole
        seconds, the streams lag differently), long enough for a car to leave the measured spot. So
        look for it: run the detector over the frames around `mid` and keep the vehicle box that
        overlaps the measured one most, or failing that lies nearest to it.
        """
        if self.detector is None:
            return None
        best, best_key = None, None
        cx, cy = (r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2
        for _, img in frames_between(self.cfg, r.camera, mid - window, mid + window):
            for d in self.detector.detect(img):
                if d.label not in VEHICLES:
                    continue
                dist = np.hypot(d.center[0] - cx, d.center[1] - cy)
                key = (iou(d.box, r.box), -dist)
                if best_key is None or key > best_key:
                    best, best_key = (img, d.box), key
        return best

    @staticmethod
    async def _safe(notify: Notifier, event: dict) -> None:
        try:
            await notify(event)
        except Exception:
            log.exception("notifier %s failed for event %d", getattr(notify, "__name__", notify), event["id"])

