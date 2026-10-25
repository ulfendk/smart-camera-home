"""Follow existing Home Assistant entities for alarm and away mode, both ways.

Many homes already have an alarm switch (an input_boolean set by a keypad, scripts and automations).
With `ha.alarm_entity` set, SmartCam arms and disarms with it, and arming or disarming SmartCam from its
own panel or web UI sets that entity, so keypads, locks and presence simulation stay in step.
`ha.away_entity` does the same for away mode. Home Assistant is the source of truth at startup.
"""

import asyncio
import json
import logging

import websockets

from ..config import Config
from ..events import EventEngine

log = logging.getLogger(__name__)

ON_STATES = {"on", "armed_away", "armed_home", "armed_night", "armed_vacation", "armed_custom_bypass",
             "triggered", "arming", "pending"}


def is_on(state: str | None) -> bool | None:
    """True/False for a known state, None for unknown/unavailable (ignored)."""
    if state in (None, "unknown", "unavailable"):
        return None
    return state in ON_STATES


def service_for(entity: str, on: bool) -> tuple[str, str]:
    domain = entity.split(".")[0]
    if domain == "alarm_control_panel":
        return domain, "alarm_arm_away" if on else "alarm_disarm"
    return domain, "turn_on" if on else "turn_off"


class HaLink:
    def __init__(self, cfg: Config, engine: EventEngine, alarm=None):
        self.cfg, self.engine, self.alarm = cfg, engine, alarm
        self.alarm_entity, self.away_entity = cfg.ha.alarm_entity, cfg.ha.away_entity
        self.ws = None
        self.n = 0
        if alarm is not None and self.alarm_entity:
            alarm.on_state.append(self._alarm_changed)
        if self.away_entity:
            engine.on_away.append(self._away_changed)
        self.ha_states: dict[str, bool | None] = {}

    @property
    def wanted(self) -> list[str]:
        return [e for e in (self.alarm_entity if self.alarm else None, self.away_entity) if e]

    async def _send(self, msg: dict) -> int:
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, **msg}))
        return self.n

    async def _apply(self, entity: str, on: bool | None) -> None:
        """Home Assistant changed: follow it."""
        if on is None or self.ha_states.get(entity) == on:
            return
        self.ha_states[entity] = on
        if entity == self.alarm_entity and self.alarm is not None:
            armed = self.alarm.state != "disarmed"
            if on and not armed:
                log.info("arming: %s turned on", entity)
                await self.alarm.arm()
            elif not on and armed:
                log.info("disarming: %s turned off", entity)
                await self.alarm.disarm()
        if entity == self.away_entity and on != self.engine.away:
            await self.engine.set_away(on)

    async def _set_ha(self, entity: str, on: bool) -> None:
        """SmartCam changed: tell Home Assistant (unless it already agrees)."""
        if self.ws is None or self.ha_states.get(entity) == on:
            return
        self.ha_states[entity] = on
        domain, service = service_for(entity, on)
        await self._send({"type": "call_service", "domain": domain, "service": service,
                          "target": {"entity_id": entity}})

    async def _alarm_changed(self, state: str) -> None:
        await self._set_ha(self.alarm_entity, state != "disarmed")

    async def _away_changed(self, away: bool) -> None:
        await self._set_ha(self.away_entity, away)

    async def run(self) -> None:
        url = self.cfg.ha.url.replace("http", "ws", 1).rstrip("/") + "/api/websocket"
        while True:
            try:
                async with websockets.connect(url, max_size=None) as ws:
                    await ws.recv()
                    await ws.send(json.dumps({"type": "auth", "access_token": self.cfg.ha.token}))
                    if json.loads(await ws.recv()).get("type") != "auth_ok":
                        raise ConnectionError("Home Assistant refused the token")
                    self.ws, self.ha_states = ws, {}
                    states_id = await self._send({"type": "get_states"})
                    await self._send({"type": "subscribe_events", "event_type": "state_changed"})
                    log.info("following %s in Home Assistant", ", ".join(self.wanted))
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("id") == states_id and msg.get("type") == "result":
                            for s in msg.get("result") or []:
                                if s["entity_id"] in self.wanted:
                                    await self._apply(s["entity_id"], is_on(s["state"]))
                        elif msg.get("type") == "event":
                            data = msg["event"]["data"]
                            if data.get("entity_id") in self.wanted and data.get("new_state"):
                                await self._apply(data["entity_id"], is_on(data["new_state"]["state"]))
            except (OSError, websockets.WebSocketException, ConnectionError) as e:
                log.warning("Home Assistant link lost (%s); retrying in 10s", e)
            self.ws = None
            await asyncio.sleep(10)
