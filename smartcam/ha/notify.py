"""Push notifications to the Home Assistant companion app through HA's REST API."""

import logging

import httpx

from ..config import Config
from ..events import EventEngine
from ..speed import SpeedConfig
from .mqtt import alert_slot

log = logging.getLogger(__name__)


def alert_payload(cfg: Config, event: dict, clip_ready: bool = False) -> dict:
    camera = event["camera"]
    reasons = ", ".join(r for r in event["reasons"] if r != "person")
    # A camera entity holding just this alert's image, so a later event doesn't change the picture.
    entity = f"camera.smartcam_alert_{alert_slot(cfg, event['id'])}"
    data = {
        # Both load through HA, so they work away from home. iOS only shows the picture via
        # entity_id (an attachment from camera_proxy is silently dropped); Android uses image.
        "entity_id": entity,
        "image": f"/api/camera_proxy/{entity}",
        "tag": f"smartcam-{event['id']}",       # the clip update replaces the first notification
        "group": "smartcam",
    }
    if cfg.ha.event_url:
        data["url"] = data["clickAction"] = cfg.ha.event_url.replace("{id}", str(event["id"]))
    return {
        "title": f"{'Clip: ' if clip_ready else ''}{event['label'].capitalize()} at {camera}",
        "message": reasons or "person detected",
        "data": data,
    }


def alarm_payload(cfg: Config, why: str, event: dict | None) -> dict:
    """The alarm push: time-sensitive (or critical) on iOS, high priority on Android."""
    sound = {"name": "default", "critical": 1, "volume": 1.0} if cfg.ha.alarm_critical else "default"
    data = {
        "tag": "smartcam-alarm",
        "push": {"interruption-level": "critical" if cfg.ha.alarm_critical else "time-sensitive", "sound": sound},
        "priority": "high", "ttl": 0, "channel": "alarm_stream",
    }
    if event is not None:
        entity = f"camera.smartcam_alert_{alert_slot(cfg, event['id'])}"
        data |= {"entity_id": entity, "image": f"/api/camera_proxy/{entity}"}
    if cfg.ha.event_url:
        data["url"] = data["clickAction"] = cfg.ha.event_url.replace("{id}", str(event["id"] if event else ""))
    return {"title": "🚨 SmartCam alarm", "message": why, "data": data}


class HaNotifier:
    def __init__(self, cfg: Config, engine: EventEngine, alarm=None):
        self.cfg = cfg
        self.http = httpx.AsyncClient(base_url=cfg.ha.url, timeout=10,
                                      headers={"Authorization": f"Bearer {cfg.ha.token}"})
        engine.on_alert.append(self.alert)
        engine.on_clip.append(self.clip)
        if alarm is not None:
            alarm.on_trigger.append(self.alarm)
        engine.on_speed.append(self.speed)

    async def _send(self, payload: dict) -> None:
        for service in self.cfg.ha.notify:
            r = await self.http.post(f"/api/services/notify/{service}", json=payload)
            r.raise_for_status()

    async def alert(self, event: dict) -> None:
        await self._send(alert_payload(self.cfg, event))
        log.info("notified event %d", event["id"])

    async def speed(self, event: dict) -> None:
        sc = SpeedConfig(**(self.cfg.cameras[event["camera"]].speed or {}))
        if sc.notify_over_kmh is None or event["speed"] < sc.notify_over_kmh:
            return
        p = alert_payload(self.cfg, event)
        limit = f" (limit {sc.limit_kmh:.0f})" if sc.limit_kmh else ""
        p["title"] = f"🚗 {event['speed']:.0f} km/h at {event['camera']}"
        plate = f" · {event['plate']}" if event.get("plate") else ""
        p["message"] = f"{event['label']}{plate}{limit}"
        await self._send(p)

    async def alarm(self, why: str, event: dict | None) -> None:
        await self._send(alarm_payload(self.cfg, why, event))
        log.info("alarm notification sent")

    async def clip(self, event: dict) -> None:
        await self._send(alert_payload(self.cfg, event, clip_ready=True))
