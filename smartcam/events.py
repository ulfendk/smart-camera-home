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

from .clips import cut_clip
from .config import Config
from .db import Database
from .pipeline import FrameResult
from .tracker import Track

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


def is_night(ts: float, rules: RulesConfig) -> bool:
    hour = datetime.fromtimestamp(ts).hour
    if rules.night_start > rules.night_end:
        return hour >= rules.night_start or hour < rules.night_end
    return rules.night_start <= hour < rules.night_end


def score_event(ev: OpenEvent, now: float, away: bool, rules: RulesConfig,
                recent_other: list[str]) -> tuple[float, list[str]]:
    score, reasons = 1.0, ["person"]
    if away:
        if ev.location == "indoor":
            return 10.0, ["person indoors while away"]
        score += 3
        reasons.append("away")
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
    return score, reasons


def should_notify(ev: OpenEvent, away: bool, now: float, rules: RulesConfig) -> bool:
    if ev.score < rules.notify_score:
        return False
    # At home only outdoor night-time events are worth a push; daytime and indoor are family.
    return away or (ev.location == "outdoor" and is_night(now, rules))


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
        self.snapshots = cfg.paths.data / "snapshots"
        self.clips = cfg.paths.data / "clips"

    @property
    def away(self) -> bool:
        return self.db.get_setting("away", "0") == "1"

    async def set_away(self, away: bool) -> None:
        self.db.set_setting("away", "1" if away else "0")
        log.info("away mode %s", "on" if away else "off")
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
        ev.score, ev.reasons = score_event(ev, now, self.away, self.rules, others)
        self.db.update_event(ev.id, score=ev.score, reasons=ev.reasons, zones=sorted(ev.zone_dwell),
                             max_score=ev.best_score)
        if not ev.notified and should_notify(ev, self.away, now, self.rules):
            ev.notified = True
            self.db.update_event(ev.id, notified=1)
            event = self.db.event(ev.id)
            for notify in self.on_alert:
                await self._safe(notify, event)

    def _snapshot(self, ev: OpenEvent, res: FrameResult, t: Track) -> None:
        img = res.frame.image.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(t.box, (w, h, w, h)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        path = self.snapshots / f"{ev.id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.db.update_event(ev.id, snapshot=str(path.relative_to(self.cfg.paths.data)))

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

    @staticmethod
    async def _safe(notify: Notifier, event: dict) -> None:
        try:
            await notify(event)
        except Exception:
            log.exception("notifier %s failed for event %d", getattr(notify, "__name__", notify), event["id"])
