"""Switch a camera's light on while a confirmed person is moving in view after dark.

"Dark" comes from Home Assistant (the sun below the horizon) when it is connected: full-colour
night cameras make the night picture nearly as bright as day, so the image cannot tell. Without
Home Assistant it is measured from the image (mean brightness) while the light is off. Once on, the
light stays on for `on_seconds` after the last sighting, and won't come back within `cooldown`.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

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


class SunDark:
    """Is the sun below the horizon? Asked of Home Assistant at most once a minute, shared by cameras."""

    def __init__(self, ha_url: str, token: str):
        import httpx
        self.http = httpx.AsyncClient(base_url=ha_url, timeout=10, headers={"Authorization": f"Bearer {token}"})
        self.value: bool | None = None
        self.checked = 0.0

    async def __call__(self) -> bool | None:
        if time.monotonic() - self.checked > 60:
            self.checked = time.monotonic()
            try:
                r = await self.http.get("/api/states/sun.sun")
                self.value = r.json().get("state") == "below_horizon" if r.status_code == 200 else None
            except Exception:
                self.value = None
        return self.value


class LightController:
    def __init__(self, cfg: LightsConfig, light, camera: str, dark_fn: Callable[[], Awaitable[bool | None]] | None = None):
        self.cfg, self.light, self.camera, self.dark_fn = cfg, light, camera, dark_fn
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
            sun = await self.dark_fn() if self.dark_fn is not None else None
            self.dark = sun if sun is not None else self.brightness(res.frame.image) < self.cfg.dark_threshold
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
