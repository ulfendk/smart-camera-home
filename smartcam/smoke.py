"""Smoke alarm: when a Home Assistant smoke sensor goes off, push a picture from inside the house.

The indoor cameras are grabbed at that moment and joined into one picture (privacy at home gives way: this is an
emergency), kept as a "smoke" event and pushed as an alarm incident: critical, to the alarm's notify services, with
the same "Confirm alarm" / "False alarm" buttons as an intrusion. It does not depend on the alarm being armed.
"""

import asyncio
import logging
import time

import cv2
import httpx
from pydantic import BaseModel

from .config import Config
from .db import Database
from .ha.mqtt import alert_slot
from .stills import grab, join

log = logging.getLogger(__name__)


class SmokeConfig(BaseModel):
    entities: list[str] = []      # smoke sensors, e.g. binary_sensor.kitchen_smoke
    cameras: list[str] = []       # joined into the picture; empty = every indoor camera
    height: int = 720             # of the joined picture
    cooldown: float = 600         # seconds before the same sensor alarms again


class Smoke:
    def __init__(self, cfg: Config, db: Database, alarm, mqtt=None, sc: SmokeConfig | None = None):
        self.cfg, self.db, self.alarm, self.mqtt = cfg, db, alarm, mqtt
        self.sc = sc or SmokeConfig(**cfg.smoke)
        self.cameras = [c for c in self.sc.cameras if c in cfg.cameras] or \
            [c.name for c in cfg.cameras.values() if c.location == "indoor"]
        self.http = httpx.AsyncClient(timeout=15)
        self.last: dict[str, float] = {}

    async def _picture(self):
        frames = [f for f in await asyncio.gather(*(grab(self.cfg, self.http, c) for c in self.cameras)) if f is not None]
        return join(frames, self.sc.height) if frames else None

    async def alarm_for(self, entity: str, name: str | None = None, test: bool = False) -> dict | None:
        """The sensor `entity` (called `name` in Home Assistant) went off. Returns the incident."""
        now = time.time()
        if not test and now - self.last.get(entity, 0) < self.sc.cooldown:
            return None
        self.last[entity] = now
        why = f"{'TEST: ' if test else ''}smoke at {name or entity}"
        log.warning("SMOKE: %s", why)
        ev_id = self.db.insert_event(camera=self.cameras[0] if self.cameras else "house", label="smoke",
                                     start_ts=now, max_score=1.0)
        self.db.update_event(ev_id, end_ts=now, reasons=["smoke", name or entity], notified=1)
        picture = await self._picture()
        if picture is not None:
            path = self.cfg.paths.data / "snapshots" / f"{ev_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.db.update_event(ev_id, snapshot=str(path.relative_to(self.cfg.paths.data)))
            if self.mqtt is not None:
                await self.mqtt._publish(f"alert/{alert_slot(self.cfg, ev_id)}", path.read_bytes())   # the push shows it
        else:
            log.warning("smoke: no camera picture")
        return await self.alarm.alert("smoke", why, ev_id)
