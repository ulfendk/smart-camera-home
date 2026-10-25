"""Power-cycle a camera whose stream has died, through a Home Assistant switch (smart plug).

Some cheap cameras lock up and only recover after losing power. A camera with `power_switch` set is
switched off and on again once its detection stream has delivered no frames for `stale_seconds`,
at most `max_per_day` times and never within `cooldown_minutes` of the last cycle.
"""

import asyncio
import logging
import time

import httpx
from pydantic import BaseModel

from .config import Config

log = logging.getLogger(__name__)


class WatchdogConfig(BaseModel):
    stale_seconds: float = 120      # no frames for this long = dead stream
    off_seconds: float = 10         # how long the power stays off
    cooldown_minutes: float = 30    # minimum time between cycles of the same camera (it also needs time to boot)
    max_per_day: int = 4            # after this, leave it alone and let a human look


class PowerWatchdog:
    def __init__(self, cfg: Config, wd: WatchdogConfig | None = None, now: float | None = None):
        self.cfg, self.wd = cfg, wd or WatchdogConfig(**cfg.watchdog)
        now = time.monotonic() if now is None else now
        self.switches = {name: cam.power_switch for name, cam in cfg.cameras.items() if cam.power_switch}
        self.last_frame = {name: now for name in self.switches}
        self.cycles: dict[str, list[float]] = {name: [] for name in self.switches}

    def seen(self, camera: str, now: float | None = None) -> None:
        if camera in self.last_frame:
            self.last_frame[camera] = time.monotonic() if now is None else now

    def due(self, now: float | None = None) -> list[str]:
        """Cameras whose stream is dead and that may be power-cycled now."""
        now = time.monotonic() if now is None else now
        out = []
        for name, last in self.last_frame.items():
            recent = [t for t in self.cycles[name] if now - t < 86400]
            self.cycles[name] = recent
            if now - last < self.wd.stale_seconds:
                continue
            if recent and now - recent[-1] < self.wd.cooldown_minutes * 60:
                continue
            if len(recent) >= self.wd.max_per_day:
                continue
            out.append(name)
        return out

    async def _switch(self, http: httpx.AsyncClient, service: str, entity: str) -> None:
        r = await http.post(f"/api/services/switch/{service}", json={"entity_id": entity})
        r.raise_for_status()

    async def cycle(self, http: httpx.AsyncClient, camera: str) -> None:
        entity = self.switches[camera]
        now = time.monotonic()
        self.cycles[camera].append(now)
        silent = now - self.last_frame[camera]
        log.warning("%s: no frames for %.0fs; power-cycling %s (%d today)",
                    camera, silent, entity, len(self.cycles[camera]))
        try:
            await self._switch(http, "turn_off", entity)
            await asyncio.sleep(self.wd.off_seconds)
            await self._switch(http, "turn_on", entity)
        except httpx.HTTPError as e:
            log.error("%s: power cycle via %s failed: %s", camera, entity, e)
        if len(self.cycles[camera]) >= self.wd.max_per_day:
            log.error("%s: power-cycled %d times in 24h; giving up until tomorrow", camera, self.wd.max_per_day)

    async def run(self) -> None:
        async with httpx.AsyncClient(base_url=self.cfg.ha.url, timeout=15,
                                     headers={"Authorization": f"Bearer {self.cfg.ha.token}"}) as http:
            log.info("power watchdog for %s", ", ".join(f"{c} ({s})" for c, s in self.switches.items()))
            while True:
                for camera in self.due():
                    await self.cycle(http, camera)
                await asyncio.sleep(15)
