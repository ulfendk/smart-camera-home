"""Push notifications to the Home Assistant companion app through HA's REST API."""

import logging

import httpx

from ..config import Config
from ..events import EventEngine
from ..speed import SpeedConfig
from ..alarm import CONFIRM, CONFIRMED, FALSE
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


ALARM_TITLES = {"intrusion": "🚨 SmartCam alarm", "smoke": "🔥 Smoke alarm"}


def alarm_payload(cfg: Config, incident: dict, event: dict | None) -> dict:
    """The alarm push: time-sensitive (or critical) on iOS, high priority on Android, with buttons to answer it."""
    sound = {"name": "default", "critical": 1, "volume": 1.0} if cfg.ha.alarm_critical else "default"
    n = incident["id"]
    data = {
        "tag": f"smartcam-alarm-{n}",
        "push": {"interruption-level": "critical" if cfg.ha.alarm_critical else "time-sensitive", "sound": sound},
        "priority": "high", "ttl": 0, "channel": "alarm_stream",
        "actions": [{"action": f"{CONFIRM}_{n}", "title": "Confirm alarm", "destructive": True},
                    {"action": f"{FALSE}_{n}", "title": "False alarm"}],
    }
    if event is not None:
        entity = f"camera.smartcam_alert_{alert_slot(cfg, event['id'])}"
        data |= {"entity_id": entity, "image": f"/api/camera_proxy/{entity}"}
    if cfg.ha.event_url:
        data["url"] = data["clickAction"] = cfg.ha.event_url.replace("{id}", str(event["id"] if event else ""))
    return {"title": ALARM_TITLES.get(incident["kind"], ALARM_TITLES["intrusion"]), "message": incident["why"], "data": data}


def response_payload(cfg: Config, incident: dict) -> dict:
    """Replaces the alarm push on every phone once someone has answered it."""
    by = f" by {incident['by']}" if incident.get("by") else ""
    confirmed = incident["status"] == CONFIRMED
    title = ALARM_TITLES.get(incident["kind"], ALARM_TITLES["intrusion"])
    data = {"tag": f"smartcam-alarm-{incident['id']}", "push": {"interruption-level": "time-sensitive"}, "priority": "high"}
    if cfg.ha.event_url and incident.get("event_id") is not None:
        data["url"] = data["clickAction"] = cfg.ha.event_url.replace("{id}", str(incident["event_id"]))
    return {"title": f"{title}: {'confirmed' if confirmed else 'false alarm'}",
            "message": f"{'Confirmed' if confirmed else 'Marked as a false alarm'}{by}. {incident['why']}", "data": data}


class HaNotifier:
    def __init__(self, cfg: Config, engine: EventEngine, alarm=None):
        self.cfg = cfg
        self.http = httpx.AsyncClient(base_url=cfg.ha.url, timeout=10,
                                      headers={"Authorization": f"Bearer {cfg.ha.token}"})
        engine.on_alert.append(self.alert)
        engine.on_clip.append(self.clip)
        if alarm is not None:
            alarm.on_trigger.append(self.alarm)
            alarm.on_response.append(self.response)
        engine.on_speed.append(self.speed)

    async def _send(self, payload: dict, services: list[str] | None = None) -> None:
        for service in services or self.cfg.ha.notify:
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

    async def alarm(self, incident: dict, event: dict | None) -> None:
        await self._send(alarm_payload(self.cfg, incident, event), self.cfg.ha.alarm_notify)
        log.info("alarm notification sent (incident %d)", incident["id"])

    async def response(self, incident: dict) -> None:
        # Home Assistant automations take it from here (escalation); the phones see who answered.
        kind = "confirmed" if incident["status"] == CONFIRMED else "false"
        await self.fire(f"smartcam_alarm_{kind}", {"incident": incident["id"], "kind": incident["kind"], "why": incident["why"],
                                                   "event_id": incident["event_id"], "by": incident["by"]})
        await self._send(response_payload(self.cfg, incident), self.cfg.ha.alarm_notify)

    async def fire(self, event_type: str, data: dict) -> None:
        """A Home Assistant event, for automations."""
        r = await self.http.post(f"/api/events/{event_type}", json=data)
        r.raise_for_status()

    async def clip(self, event: dict) -> None:
        await self._send(alert_payload(self.cfg, event, clip_ready=True))
