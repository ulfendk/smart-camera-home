"""A doorbell picture: when a Home Assistant entity turns on (the bell's switch or binary sensor),
grab the chosen cameras at that moment, join them side by side into one picture (iOS notifications
carry one), publish it as camera.smartcam_doorbell, keep it as a "doorbell" event and send the push.
"""

import asyncio
import json
import logging
import time

import cv2
import httpx
import numpy as np
import websockets
from pydantic import BaseModel

from ..config import Config
from ..db import Database

log = logging.getLogger(__name__)


class DoorbellConfig(BaseModel):
    entity: str | None = None               # e.g. switch.doorbell_channel_1
    cameras: list[str] = []                 # joined side by side, in this order
    notify: list[str] = []                  # notify services; empty = ha.notify
    title: str = "Doorbell"
    message: str = "Someone is at the door"
    height: int = 720                       # of the joined picture


def join(images: list[np.ndarray], height: int) -> np.ndarray:
    scaled = [cv2.resize(i, (round(i.shape[1] * height / i.shape[0]), height), interpolation=cv2.INTER_AREA) for i in images]
    gap = np.zeros((height, 6, 3), np.uint8)
    out = scaled[0]
    for img in scaled[1:]:
        out = np.hstack([out, gap, img])
    return out


class Doorbell:
    def __init__(self, cfg: Config, db: Database, mqtt=None, dc: DoorbellConfig | None = None):
        self.cfg, self.db, self.mqtt = cfg, db, mqtt
        self.dc = dc or DoorbellConfig(**cfg.doorbell)
        self.http = httpx.AsyncClient(timeout=15)
        self.ha = httpx.AsyncClient(base_url=cfg.ha.url, timeout=15, headers={"Authorization": f"Bearer {cfg.ha.token}"})
        self.last = 0.0

    async def _frame(self, camera: str) -> np.ndarray | None:
        r = await self.http.get(f"http://127.0.0.1:{self.cfg.go2rtc.api_port}/api/frame.jpeg", params={"src": camera})
        if r.status_code != 200:
            return None
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)

    async def ring(self) -> int | None:
        """Take the picture and tell everyone. Returns the event id."""
        if time.time() - self.last < 10:
            return None   # one ring, not a burst of presses
        self.last = now = time.time()
        frames = [f for f in await asyncio.gather(*(self._frame(c) for c in self.dc.cameras)) if f is not None]
        if not frames:
            log.warning("doorbell: no camera picture")
            return None
        picture = join(frames, self.dc.height)
        ev_id = self.db.insert_event(camera=self.dc.cameras[0], label="doorbell", start_ts=now, max_score=1.0)
        path = self.cfg.paths.data / "snapshots" / f"{ev_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.db.update_event(ev_id, end_ts=now, snapshot=str(path.relative_to(self.cfg.paths.data)), reasons=["doorbell"])
        if self.mqtt is not None:
            await self.mqtt._publish("doorbell/snapshot", path.read_bytes())
        await self._push()
        log.info("doorbell: event %d (%s)", ev_id, ", ".join(self.dc.cameras))
        return ev_id

    async def _push(self) -> None:
        entity = "camera.smartcam_doorbell"
        payload = {"title": self.dc.title, "message": self.dc.message, "data": {
            "entity_id": entity, "image": f"/api/camera_proxy/{entity}", "tag": "smartcam-doorbell",
            "push": {"interruption-level": "time-sensitive"}, "priority": "high", "ttl": 0}}
        for service in self.dc.notify or self.cfg.ha.notify:
            try:
                await self.ha.post(f"/api/services/notify/{service}", json=payload)
            except httpx.HTTPError as e:
                log.warning("doorbell push to %s failed: %s", service, e)

    async def run(self) -> None:
        url = self.cfg.ha.url.replace("http", "ws", 1).rstrip("/") + "/api/websocket"
        while True:
            try:
                async with websockets.connect(url, max_size=None) as ws:
                    await ws.recv()
                    await ws.send(json.dumps({"type": "auth", "access_token": self.cfg.ha.token}))
                    await ws.recv()
                    await ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "state_changed"}))
                    log.info("doorbell: watching %s", self.dc.entity)
                    async for raw in ws:
                        msg = json.loads(raw)
                        data = msg.get("event", {}).get("data", {}) if msg.get("type") == "event" else {}
                        new, old = data.get("new_state") or {}, data.get("old_state") or {}
                        if data.get("entity_id") == self.dc.entity and new.get("state") == "on" and old.get("state") != "on":
                            asyncio.create_task(self.ring())
            except (OSError, websockets.WebSocketException) as e:
                log.warning("doorbell: Home Assistant connection lost (%s); retrying", e)
            await asyncio.sleep(10)
