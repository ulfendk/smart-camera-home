"""Restart a camera whose stream has died.

Some cameras lock up and only recover after a restart. Once a camera's detection stream has delivered
no frames for `stale_seconds`, it is restarted, at most `max_per_day` times and never within
`cooldown_minutes` of the last restart:
- with `power_switch` set: the Home Assistant switch (a smart plug) is switched off and on again
- otherwise a Dahua camera is rebooted through its own HTTP API
"""

import asyncio
import logging
import time
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from .config import Config

log = logging.getLogger(__name__)


class WatchdogConfig(BaseModel):
    enabled: bool = True
    stale_seconds: float = 120      # no frames for this long = dead stream
    off_seconds: float = 10         # how long the power stays off
    cooldown_minutes: float = 30    # minimum time between cycles of the same camera (it also needs time to boot)
    max_per_day: int = 4            # after this, leave it alone and let a human look


class PowerWatchdog:
    def __init__(self, cfg: Config, wd: WatchdogConfig | None = None, now: float | None = None):
        self.cfg, self.wd = cfg, wd or WatchdogConfig(**cfg.watchdog)
        now = time.monotonic() if now is None else now
        on = self.wd.enabled
        self.switches = {name: cam.power_switch for name, cam in cfg.cameras.items()
                         if on and cam.power_switch and cfg.ha.enabled}
        self.dahua = {name: cam for name, cam in cfg.cameras.items()
                      if on and cam.vendor == "dahua" and name not in self.switches}
        watched = [*self.switches, *self.dahua]
        self.last_frame = {name: now for name in watched}
        self.cycles: dict[str, list[float]] = {name: [] for name in watched}

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

    async def _reboot_dahua(self, camera: str) -> None:
        cam = self.dahua[camera]
        url = urlsplit(cam.record)
        async with httpx.AsyncClient(auth=httpx.DigestAuth(url.username, url.password), timeout=10) as http:
            r = await http.get(f"http://{cam.ip}/cgi-bin/magicBox.cgi?action=reboot")
            r.raise_for_status()
            if "OK" not in r.text:
                raise httpx.HTTPError(f"unexpected reply: {r.text.strip()[:100]}")

    async def cycle(self, ha: httpx.AsyncClient | None, camera: str) -> None:
        now = time.monotonic()
        self.cycles[camera].append(now)
        silent = now - self.last_frame[camera]
        how = f"power-cycling {self.switches[camera]}" if camera in self.switches else "rebooting it"
        log.warning("%s: no frames for %.0fs; %s (%d today)", camera, silent, how, len(self.cycles[camera]))
        try:
            if camera in self.switches:
                await self._switch(ha, "turn_off", self.switches[camera])
                await asyncio.sleep(self.wd.off_seconds)
                await self._switch(ha, "turn_on", self.switches[camera])
            else:
                await self._reboot_dahua(camera)
        except httpx.HTTPError as e:
            log.error("%s: restart failed: %s", camera, e)
        if len(self.cycles[camera]) >= self.wd.max_per_day:
            log.error("%s: power-cycled %d times in 24h; giving up until tomorrow", camera, self.wd.max_per_day)

    async def run(self) -> None:
        async with httpx.AsyncClient(base_url=self.cfg.ha.url, timeout=15,
                                     headers={"Authorization": f"Bearer {self.cfg.ha.token}"}) as ha:
            log.info("stream watchdog: %s", ", ".join(
                [f"{c} (power {s})" for c, s in self.switches.items()] + [f"{c} (reboot)" for c in self.dahua]))
            while True:
                for camera in self.due():
                    await self.cycle(ha, camera)
                await asyncio.sleep(15)
