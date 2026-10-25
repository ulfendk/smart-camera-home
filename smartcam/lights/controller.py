"""Switch a camera's light on while a confirmed person is moving in view after dark.

"Dark" is measured from the image (mean brightness) while the light is off, so it follows real
light conditions (early winter dusk, floodlit driveway) instead of a clock. Once on, the light
stays on for `on_seconds` after the last sighting, and won't come back within `cooldown`.
"""

import asyncio
import logging

import cv2
from pydantic import BaseModel

from ..pipeline import FrameResult

log = logging.getLogger(__name__)


class LightsConfig(BaseModel):
    enabled: bool = False
    dry_run: bool = True
    cameras: list[str] = []       # cameras whose own light is used; empty = all outdoor Dahuas
    on_seconds: float = 60
    max_on_seconds: float = 600
    cooldown: float = 20
    dark_threshold: float = 70    # mean luma (0-255) below which the scene counts as dark


class LightController:
    def __init__(self, cfg: LightsConfig, light, camera: str):
        self.cfg, self.light, self.camera = cfg, light, camera
        self.dark = False
        self.on_since: float | None = None
        self.last_seen = 0.0
        self.off_at = float("-inf")

    @staticmethod
    def brightness(image) -> float:
        small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
        return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

    async def handle(self, res: FrameResult, confirmed_ids: set[int]) -> None:
        now = res.frame.ts
        if self.on_since is None:
            self.dark = self.brightness(res.frame.image) < self.cfg.dark_threshold
        person = any(t.id in confirmed_ids and t.label == "person" and t.last_ts == now and t.moving
                     for t in res.tracks)
        if person:
            self.last_seen = now

        if self.on_since is None:
            if person and self.dark and now - self.off_at >= self.cfg.cooldown:
                await self._switch(True, now)
        elif now - self.last_seen > self.cfg.on_seconds or now - self.on_since > self.cfg.max_on_seconds:
            await self._switch(False, now)

    async def _switch(self, on: bool, now: float) -> None:
        try:
            await (self.light.on() if on else self.light.off())
        except Exception:
            log.exception("light %s: switching %s failed", self.camera, "on" if on else "off")
            return
        if on:
            self.on_since = now
        else:
            self.on_since, self.off_at = None, now
