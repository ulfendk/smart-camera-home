"""Home Assistant lights up to full while a person is outside after dark.

Each camera can list `person_lights`. When SmartCam confirms a person on that camera and it is dark,
those lights go to `brightness` %; `seconds` after the person has gone they return to exactly what
they were before (off, or the dimmed evening level). A light shared by several cameras stays up until
all of them are quiet, and a light someone changes by hand meanwhile is left as it is.

Dark: a numeric sensor (`dark_entity`, e.g. a lux sensor) below `dark_below`, else the sun below
the horizon.
"""

import asyncio
import logging

import httpx
from pydantic import BaseModel

from ..config import Config
from ..events import EventEngine

log = logging.getLogger(__name__)


class PersonLightsConfig(BaseModel):
    enabled: bool = True
    brightness: int = 100           # percent
    seconds: float = 120            # after the person has gone
    dark_entity: str | None = None  # e.g. a lux sensor; default: sun.sun below the horizon
    dark_below: float = 50


class PersonLights:
    def __init__(self, cfg: Config, engine: EventEngine, plc: PersonLightsConfig | None = None, http=None):
        self.cfg = cfg
        self.plc = plc or PersonLightsConfig(**cfg.person_lights)
        self.map = {name: cam.person_lights for name, cam in cfg.cameras.items() if cam.person_lights}
        self.http = http or httpx.AsyncClient(base_url=cfg.ha.url, timeout=15,
                                              headers={"Authorization": f"Bearer {cfg.ha.token}"})
        self.active: set[str] = set()                 # cameras with a person right now
        self.before: dict[str, dict] = {}             # light -> its state before we raised it
        self.timers: dict[str, asyncio.Task] = {}     # camera -> pending restore
        engine.on_open.append(self._person)
        engine.on_close.append(self._gone)

    async def _state(self, entity: str) -> dict | None:
        r = await self.http.get(f"/api/states/{entity}")
        return r.json() if r.status_code == 200 else None

    async def dark(self) -> bool:
        if self.plc.dark_entity:
            s = await self._state(self.plc.dark_entity)
            try:
                return float(s["state"]) < self.plc.dark_below
            except (TypeError, KeyError, ValueError):
                pass   # sensor missing or not a number: fall back to the sun
        s = await self._state("sun.sun")
        return s is not None and s["state"] == "below_horizon"

    async def _person(self, event: dict) -> None:
        cam = event["camera"]
        if cam not in self.map or event["label"] != "person":
            return
        if (t := self.timers.pop(cam, None)) is not None:
            t.cancel()
        self.active.add(cam)
        if not await self.dark():
            return
        for light in self.map[cam]:
            if light in self.before:
                continue   # already up (another camera)
            s = await self._state(light)
            if s is None or s["state"] == "unavailable":
                continue
            self.before[light] = {"on": s["state"] == "on", "brightness": s["attributes"].get("brightness")}
            await self.http.post("/api/services/light/turn_on",
                                 json={"entity_id": light, "brightness_pct": self.plc.brightness})
        log.info("person on %s after dark: %s up", cam, ", ".join(self.map[cam]))

    async def _gone(self, event: dict) -> None:
        cam = event["camera"]
        if cam in self.map and event["label"] == "person":
            self.timers[cam] = asyncio.create_task(self._restore_later(cam))

    async def _restore_later(self, cam: str) -> None:
        await asyncio.sleep(self.plc.seconds)
        self.active.discard(cam)
        self.timers.pop(cam, None)
        still_needed = {l for c in self.active for l in self.map[c]}
        for light in self.map[cam]:
            if light in still_needed or light not in self.before:
                continue
            was = self.before.pop(light)
            s = await self._state(light)
            if s is None or s["attributes"].get("brightness") not in (None, round(self.plc.brightness * 2.55)):
                continue   # changed by hand meanwhile: leave it
            if was["on"]:
                data = {"entity_id": light} | ({"brightness": was["brightness"]} if was["brightness"] else {})
                await self.http.post("/api/services/light/turn_on", json=data)
            else:
                await self.http.post("/api/services/light/turn_off", json={"entity_id": light})
        log.info("%s quiet: lights back as they were", cam)
