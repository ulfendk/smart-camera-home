"""Alarm mode: arm from Home Assistant; when a serious event happens while armed, deter.

States follow Home Assistant's alarm panel: disarmed, armed_away, triggered. Arming also switches on
away mode (and disarming switches it off), so scoring and notifications follow the alarm.

When triggered: the outdoor Dahuas' red/blue warning lights come on for `light_minutes`, and, if
`siren` is on, the sirens sound in rounds of `siren_on_seconds` with `siren_off_seconds` pauses. After
that the alarm goes quiet and does not trigger again until it is disarmed and armed again.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from pydantic import BaseModel

from .config import Config
from .db import Database
from .events import EventEngine, OpenEvent
from .lights.dahua_rpc import SIREN, WARNING_LIGHT, DahuaRpc

log = logging.getLogger(__name__)

DISARMED, ARMED, TRIGGERED = "disarmed", "armed_away", "triggered"


class AlarmConfig(BaseModel):
    enabled: bool = True
    trigger_score: float = 8        # event score that triggers the alarm while armed (indoor person = 10)
    cameras: list[str] = []         # cameras whose deterrence is used; empty = every outdoor Dahua
    light_minutes: float = 15       # red/blue warning lights
    siren: bool = False             # many cameras refuse siren commands; switch on once tested
    siren_on_seconds: float = 10
    siren_off_seconds: float = 20
    siren_rounds: int = 3


class Alarm:
    def __init__(self, cfg: Config, db: Database, engine: EventEngine, ac: AlarmConfig | None = None, outputs=None):
        self.cfg, self.db, self.engine = cfg, db, engine
        self.ac = ac or AlarmConfig(**cfg.alarm)
        names = self.ac.cameras or [c.name for c in cfg.cameras.values() if c.vendor == "dahua" and c.location == "outdoor"]
        # camera -> object with async output(kind, on); injectable for tests
        self.outputs = outputs if outputs is not None else {n: _rpc_for(cfg, n) for n in names if n in cfg.cameras}
        self.on_state: list[Callable[[str], Awaitable[None]]] = []
        self.spent = self.db.get_setting("alarm_spent", "0") == "1"   # triggered once since arming
        self._task: asyncio.Task | None = None
        engine.on_score.append(self._scored)

    @property
    def state(self) -> str:
        return self.db.get_setting("alarm_state", DISARMED)

    async def _set_state(self, state: str) -> None:
        self.db.set_setting("alarm_state", state)
        log.info("alarm %s", state)
        for cb in self.on_state:
            try:
                await cb(state)
            except Exception:
                log.exception("alarm state callback failed")

    async def arm(self) -> None:
        self.spent = False
        self.db.set_setting("alarm_spent", "0")
        await self._set_state(ARMED)
        if not self.engine.away:
            await self.engine.set_away(True)

    async def disarm(self) -> None:
        if self._task:
            self._task.cancel()
        await self._set_state(DISARMED)
        if self.engine.away:
            await self.engine.set_away(False)

    async def command(self, cmd: str) -> None:
        """Home Assistant alarm panel commands."""
        cmd = cmd.upper()
        if cmd in ("ARM_AWAY", "ARM_HOME", "ARM_NIGHT", "ARM_VACATION", "ARM_CUSTOM_BYPASS"):
            await self.arm()
        elif cmd == "DISARM":
            await self.disarm()

    async def _scored(self, ev: OpenEvent) -> None:
        if self.ac.enabled and self.state == ARMED and not self.spent and ev.score >= self.ac.trigger_score:
            await self.trigger(f"{ev.camera}: {', '.join(ev.reasons)}")

    async def trigger(self, why: str = "manual") -> None:
        self.spent = True
        self.db.set_setting("alarm_spent", "1")
        log.warning("ALARM triggered (%s)", why)
        await self._set_state(TRIGGERED)
        self._task = asyncio.create_task(self._deter())

    async def _all(self, kind: int, on: bool) -> None:
        async def one(name, out):
            try:
                await out.output(kind, on)
            except Exception as e:
                log.warning("alarm: %s %s %s failed: %s", name, "siren" if kind == SIREN else "light", "on" if on else "off", e)
        await asyncio.gather(*(one(n, o) for n, o in self.outputs.items()))

    async def _deter(self) -> None:
        light_end = asyncio.get_running_loop().time() + self.ac.light_minutes * 60
        try:
            await self._all(WARNING_LIGHT, True)
            if self.ac.siren:
                for _ in range(self.ac.siren_rounds):
                    await self._all(SIREN, True)
                    await asyncio.sleep(self.ac.siren_on_seconds)
                    await self._all(SIREN, False)
                    await asyncio.sleep(self.ac.siren_off_seconds)
            while (left := light_end - asyncio.get_running_loop().time()) > 0:
                await asyncio.sleep(min(left, 60))
                if light_end - asyncio.get_running_loop().time() > 0:
                    await self._all(WARNING_LIGHT, True)   # refresh, in case a camera dropped it
        finally:
            await asyncio.shield(self._quiet())
            if self.state == TRIGGERED:
                await self._set_state(ARMED)   # armed, but spent until disarmed and armed again

    async def _quiet(self) -> None:
        if self.ac.siren:
            await self._all(SIREN, False)
        await self._all(WARNING_LIGHT, False)

    async def startup(self) -> None:
        """After a restart in the middle of an alarm: make sure everything is off, stay armed (spent)."""
        if self.state == TRIGGERED:
            await self._quiet()
            await self._set_state(ARMED)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def _rpc_for(cfg: Config, name: str) -> DahuaRpc:
    cam = cfg.cameras[name]
    url = urlsplit(cam.record)
    return DahuaRpc(cam.ip, url.username, url.password)
