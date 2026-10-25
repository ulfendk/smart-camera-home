"""Home Assistant via MQTT discovery.

Entities:
- switch.smartcam_away                  away mode (also settable from the smartcam UI)
- binary_sensor.smartcam_<camera>_person on while a confirmed person event is open on that camera
- camera.smartcam_<camera>_snapshot     latest event snapshot; notifications use it as their image,
                                        so it loads through HA even when the phone is not at home
"""

import asyncio
import json
import logging
from pathlib import Path

import aiomqtt

from ..config import Config
from ..events import EventEngine

log = logging.getLogger(__name__)


def alert_slot(cfg: Config, event_id: int) -> int:
    """The alert-image camera entity an event uses; it keeps that image until the slot comes round again."""
    return event_id % cfg.mqtt.alert_slots + 1


def discovery_messages(cfg: Config) -> list[tuple[str, dict]]:
    p, d = cfg.mqtt.topic_prefix, cfg.mqtt.discovery_prefix
    device = {"identifiers": ["smartcam"], "name": "Smartcam", "manufacturer": "smart-camera-home"}
    avail = {"availability_topic": f"{p}/available"}
    msgs = [(f"{d}/switch/smartcam/away/config", {
        "name": "Away", "unique_id": "smartcam_away", "default_entity_id": "switch.smartcam_away", "icon": "mdi:home-export-outline",
        "state_topic": f"{p}/away/state", "command_topic": f"{p}/away/set", "device": device, **avail})]
    for name in cfg.cameras:
        msgs.append((f"{d}/binary_sensor/smartcam/{name}_person/config", {
            "name": f"{name} person", "unique_id": f"smartcam_{name}_person", "default_entity_id": f"binary_sensor.smartcam_{name}_person",
            "device_class": "occupancy", "state_topic": f"{p}/{name}/person", "device": device, **avail}))
        msgs.append((f"{d}/camera/smartcam/{name}_snapshot/config", {
            "name": f"{name} snapshot", "unique_id": f"smartcam_{name}_snapshot",
            "default_entity_id": f"camera.smartcam_{name}_snapshot", "topic": f"{p}/{name}/snapshot", "device": device, **avail}))
    for n in range(1, cfg.mqtt.alert_slots + 1):
        msgs.append((f"{d}/camera/smartcam/alert_{n}/config", {
            "name": f"alert {n}", "unique_id": f"smartcam_alert_{n}", "default_entity_id": f"camera.smartcam_alert_{n}",
            "entity_category": "diagnostic", "topic": f"{p}/alert/{n}", "device": device, **avail}))
    return msgs


class HaMqtt:
    def __init__(self, cfg: Config, engine: EventEngine):
        self.cfg, self.engine = cfg, engine
        self.client: aiomqtt.Client | None = None
        self.open_events: dict[str, set[int]] = {name: set() for name in cfg.cameras}
        engine.on_open.append(self._event_open)
        engine.on_close.append(self._event_close)
        # Before the notifiers, which point at these entities.
        engine.on_alert.insert(0, self._publish_alert_image)
        engine.on_clip.insert(0, self._publish_alert_image)
        engine.on_away.append(self._publish_away)

    async def run(self) -> None:
        m, p = self.cfg.mqtt, self.cfg.mqtt.topic_prefix
        will = aiomqtt.Will(f"{p}/available", "offline", retain=True)
        while True:
            try:
                async with aiomqtt.Client(m.host, m.port, username=m.user, password=m.password,
                                          identifier="smartcam", will=will) as client:
                    self.client = client
                    for topic, payload in discovery_messages(self.cfg):
                        await client.publish(topic, json.dumps(payload), retain=True)
                    await client.publish(f"{p}/available", "online", retain=True)
                    await self._publish_away(self.engine.away)
                    for name in self.cfg.cameras:
                        await client.publish(f"{p}/{name}/person", "OFF", retain=True)
                    await client.subscribe(f"{p}/away/set")
                    log.info("connected to MQTT %s:%d", m.host, m.port)
                    async for msg in client.messages:
                        await self.engine.set_away(msg.payload.decode().upper() == "ON")
            except aiomqtt.MqttError as e:
                self.client = None
                log.warning("MQTT connection lost (%s); retrying in 10s", e)
                await asyncio.sleep(10)

    async def _publish(self, topic: str, payload, retain: bool = True) -> None:
        if self.client is not None:
            await self.client.publish(f"{self.cfg.mqtt.topic_prefix}/{topic}", payload, retain=retain)

    async def _publish_away(self, away: bool) -> None:
        await self._publish("away/state", "ON" if away else "OFF")

    async def _event_open(self, event: dict) -> None:
        self.open_events[event["camera"]].add(event["id"])
        await self._publish(f"{event['camera']}/person", "ON")
        await self._publish_snapshot(event)   # the snapshot entity always shows the latest event

    async def _event_close(self, event: dict) -> None:
        await self._publish_snapshot(event)   # the best frame of the whole event
        self.open_events[event["camera"]].discard(event["id"])
        if not self.open_events[event["camera"]]:
            await self._publish(f"{event['camera']}/person", "OFF")

    async def _publish_alert_image(self, event: dict) -> None:
        await self._publish_snapshot(event)
        if event.get("snapshot"):
            data = (self.cfg.paths.data / event["snapshot"]).read_bytes()
            await self._publish(f"alert/{alert_slot(self.cfg, event['id'])}", data)

    async def _publish_snapshot(self, event: dict) -> None:
        if event.get("snapshot"):
            data = (self.cfg.paths.data / event["snapshot"]).read_bytes()
            await self._publish(f"{event['camera']}/snapshot", data)
