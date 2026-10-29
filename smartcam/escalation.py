"""Escalation: once someone confirms an alarm (the "Confirm alarm" button), the house makes itself heard.

Every chosen light goes to full brightness and the speakers play a siren, the way a doorbell automation plays its
ding-dong (media_player.play_media). The siren is SmartCam's own (generated, served at /siren.wav), so nothing
needs copying into Home Assistant. Before anything changes, the lights are saved as a Home Assistant scene and the
Sonos speakers snapshotted; stopping (the "Stop siren" button, disarming, or the time running out) puts both back.
"""

import asyncio
import io
import logging
import wave

import httpx
import numpy as np
from pydantic import BaseModel

from .config import Config

log = logging.getLogger(__name__)

SCENE = "scene.smartcam_before_alarm"
SIREN_SECONDS = 30   # one play of the siren; it is played again until stopped


class EscalationConfig(BaseModel):
    enabled: bool = False
    lights: list[str] = ["all"]      # to full brightness; "all" = every light in Home Assistant
    speakers: list[str] = []         # media players for the siren, e.g. Sonos speakers
    volume: float = 0.6              # speaker volume during the siren (0..1)
    siren_minutes: float = 5         # the siren stops after this long
    lights_minutes: float = 30       # the lights go back after this long (or when stopped / disarmed)


def siren(seconds: float = SIREN_SECONDS, rate: int = 22050) -> bytes:
    """A wailing siren as a WAV file: a tone sweeping 600-1400 Hz and back every 3 s, clipped for loudness."""
    t = np.arange(int(seconds * rate)) / rate
    freq = 1000 - 400 * np.cos(2 * np.pi * t / 3)
    phase = 2 * np.pi * np.cumsum(freq) / rate
    wave_ = np.clip(1.6 * np.sin(phase), -1, 1) * 0.8
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((wave_ * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class Escalation:
    def __init__(self, cfg: Config, alarm, ec: EscalationConfig | None = None, http: httpx.AsyncClient | None = None):
        self.cfg, self.alarm = cfg, alarm
        self.ec = ec or EscalationConfig(**cfg.escalation)
        self.http = http or httpx.AsyncClient(base_url=cfg.ha.url, timeout=15,
                                              headers={"Authorization": f"Bearer {cfg.ha.token}"})
        self.active = False
        self._siren: asyncio.Task | None = None
        self._lights: asyncio.Task | None = None
        alarm.on_response.append(self._answered)
        alarm.on_state.append(self._state)
        alarm.on_stop.append(lambda incident: self.stop())

    @property
    def siren_url(self) -> str:
        return f"http://{self.cfg.ha.smartcam_host}:{self.cfg.web.port}/siren.wav"

    async def _call(self, service: str, data: dict) -> bool:
        domain, name = service.split(".")
        try:
            r = await self.http.post(f"/api/services/{domain}/{name}", json=data)
            r.raise_for_status()
            return True
        except httpx.HTTPError as e:
            log.warning("escalation: %s failed: %s", service, e)
            return False

    async def _each(self, service: str, data: dict) -> None:
        """Per speaker: one that is unavailable (or not a Sonos) must not fail the others."""
        await asyncio.gather(*(self._call(service, {"entity_id": s, **data}) for s in self.ec.speakers))

    async def _answered(self, incident: dict) -> None:
        if incident["status"] == "confirmed":
            await self.start(incident)

    async def _state(self, state: str) -> None:
        if state == "disarmed" and self.active:
            await self.stop()

    async def _light_ids(self) -> list[str]:
        if self.ec.lights != ["all"]:
            return self.ec.lights
        r = await self.http.get("/api/states")
        r.raise_for_status()
        return [s["entity_id"] for s in r.json() if s["entity_id"].startswith("light.") and s["state"] in ("on", "off")]

    async def start(self, incident: dict | None = None, siren_minutes: float | None = None,
                    lights_minutes: float | None = None) -> None:
        if self.active:
            return
        self.active = True
        log.warning("escalation: lights to full, siren on %s", ", ".join(self.ec.speakers) or "no speakers")
        if self.ec.lights:
            try:
                lights = await self._light_ids()
            except httpx.HTTPError as e:
                log.warning("escalation: no light list (%s)", e)
                lights = []
            if lights:
                await self._call("scene.create", {"scene_id": SCENE.split(".")[1], "snapshot_entities": lights})
                await self._call("light.turn_on", {"entity_id": "all" if self.ec.lights == ["all"] else lights,
                                                   "brightness_pct": 100})
            self._lights = asyncio.create_task(self._later(lights_minutes or self.ec.lights_minutes, self.stop))
        if self.ec.speakers:
            await self._each("sonos.snapshot", {"with_group": True})
            self._siren = asyncio.create_task(self._play(siren_minutes or self.ec.siren_minutes))

    async def _later(self, minutes: float, then) -> None:
        await asyncio.sleep(minutes * 60)
        await then()

    async def _play(self, minutes: float) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + minutes * 60
        try:
            await self._each("media_player.volume_set", {"volume_level": self.ec.volume})
            while (left := end - loop.time()) > 0:
                await self._each("media_player.play_media", {"media_content_id": self.siren_url, "media_content_type": "music"})
                await asyncio.sleep(min(left, SIREN_SECONDS))
        finally:
            await asyncio.shield(self._quiet())

    async def _quiet(self) -> None:
        await self._each("media_player.media_stop", {})
        await self._each("sonos.restore", {"with_group": True})

    async def stop(self) -> None:
        """Siren off and the lights back as they were."""
        if not self.active:
            return
        self.active = False
        log.info("escalation: stopped, putting lights and speakers back")
        current = asyncio.current_task()
        for task in (self._siren, self._lights):
            if task and task is not current and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._siren = self._lights = None
        if self.ec.lights:
            await self._call("scene.turn_on", {"entity_id": SCENE})
