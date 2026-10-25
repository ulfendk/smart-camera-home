"""Push notifications to the Home Assistant companion app through HA's REST API."""

import logging

import httpx

from ..config import Config
from ..events import EventEngine
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


class HaNotifier:
    def __init__(self, cfg: Config, engine: EventEngine):
        self.cfg = cfg
        self.http = httpx.AsyncClient(base_url=cfg.ha.url, timeout=10,
                                      headers={"Authorization": f"Bearer {cfg.ha.token}"})
        engine.on_alert.append(self.alert)
        engine.on_clip.append(self.clip)

    async def _send(self, payload: dict) -> None:
        for service in self.cfg.ha.notify:
            r = await self.http.post(f"/api/services/notify/{service}", json=payload)
            r.raise_for_status()

    async def alert(self, event: dict) -> None:
        await self._send(alert_payload(self.cfg, event))
        log.info("notified event %d", event["id"])

    async def clip(self, event: dict) -> None:
        await self._send(alert_payload(self.cfg, event, clip_ready=True))
