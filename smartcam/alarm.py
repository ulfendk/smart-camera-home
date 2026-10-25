"""Alarm mode: arm from Home Assistant; when a serious event happens while armed, deter.

States follow Home Assistant's alarm panel: disarmed, arming, armed_away, pending, triggered.

Arming waits `exit_delay` seconds ("arming") before the alarm and away mode become active, and only
events that start after that can trigger, so nobody sets it off on the way out. A trigger first waits
`entry_delay` seconds ("pending"), so coming home and disarming at the keypad cancels it. Disarming
switches away mode off.

When triggered: the outdoor Dahuas' red/blue warning lights come on for `light_minutes`, and, if
`siren` is on, the sirens sound in rounds of `siren_on_seconds` with `siren_off_seconds` pauses. After
that the alarm goes quiet and does not trigger again until it is disarmed and armed again.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from pydantic import BaseModel

from .config import Config
from .db import Database
from .events import EventEngine, OpenEvent
from .lights.dahua_rpc import SIREN, WARNING_LIGHT, DahuaRpc

log = logging.getLogger(__name__)

DISARMED, ARMING, ARMED, PENDING, TRIGGERED = "disarmed", "arming", "armed_away", "pending", "triggered"


class AlarmConfig(BaseModel):
    enabled: bool = True
    trigger_score: float = 8        # event score that triggers the alarm while armed (indoor person = 10)
    exit_delay: float = 60          # seconds after arming before it is active (time to leave)
    entry_delay: float = 30         # seconds between a trigger and the alarm going off (time to disarm)
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
        self.on_trigger: list[Callable[[str, dict | None], Awaitable[None]]] = []   # (why, event)
        self.spent = self.db.get_setting("alarm_spent", "0") == "1"   # triggered once since arming
        self._task: asyncio.Task | None = None        # exit delay, entry delay or deterrence
        self.armed_at = float(self.db.get_setting("alarm_armed_at", "0"))   # when it became active
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
        if self.state in (ARMING, ARMED, PENDING, TRIGGERED):
            return
        self.spent = False
        self.db.set_setting("alarm_spent", "0")
        if self.ac.exit_delay > 0:
            await self._set_state(ARMING)
            self._task = asyncio.create_task(self._after_exit_delay())
        else:
            await self._activate()

    async def _after_exit_delay(self) -> None:
        await asyncio.sleep(self.ac.exit_delay)
        if self.state == ARMING:
            await self._activate()

    async def _activate(self) -> None:
        self.armed_at = time.time()
        self.db.set_setting("alarm_armed_at", str(self.armed_at))
        await self._set_state(ARMED)
        if not self.engine.away:
            await self.engine.set_away(True)

    async def disarm(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
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
        # Only activity that started after the alarm became active counts: not the people leaving.
        if (self.ac.enabled and self.state == ARMED and not self.spent and ev.start >= self.armed_at
                and ev.score >= self.ac.trigger_score):
            await self.trigger(f"{ev.camera}: {', '.join(r for r in ev.reasons if r != 'person')}", ev.id)

    async def trigger(self, why: str = "manual", event_id: int | None = None, delay: float | None = None) -> None:
        self.spent = True
        self.db.set_setting("alarm_spent", "1")
        delay = self.ac.entry_delay if delay is None else delay
        if delay > 0:
            log.warning("alarm pending for %.0fs (%s)", delay, why)
            await self._set_state(PENDING)
            self._task = asyncio.create_task(self._after_entry_delay(why, event_id, delay))
        else:
            await self._go_off(why, event_id)

    async def _after_entry_delay(self, why: str, event_id: int | None, delay: float) -> None:
        await asyncio.sleep(delay)
        if self.state == PENDING:
            await self._go_off(why, event_id)

    async def _go_off(self, why: str, event_id: int | None) -> None:
        log.warning("ALARM triggered (%s)", why)
        await self._set_state(TRIGGERED)
        self._task = asyncio.create_task(self._deter())
        event = self.db.event(event_id) if event_id is not None else None
        for cb in self.on_trigger:
            try:
                await cb(why, event)
            except Exception:
                log.exception("alarm trigger callback failed")

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
        """After a restart: an alarm in progress is switched off (staying armed, spent); delays finish at once."""
        if self.state == TRIGGERED:
            await self._quiet()
            await self._set_state(ARMED)
        elif self.state == PENDING:
            await self._set_state(ARMED)
        elif self.state == ARMING:
            await self._activate()

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
