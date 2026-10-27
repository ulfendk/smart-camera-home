"""Live camera entities in Home Assistant, through its Generic Camera integration.

Each SmartCam camera becomes `camera.smartcam_<name>`: the stream is SmartCam's go2rtc restream (shared
on the LAN with a password, `go2rtc.rtsp_lan`), the still image comes through the web UI's player proxy.
Home Assistant then serves the stream itself (WebRTC in dashboards, also remotely). Cameras that
already have an entry are left alone, so this can be run again after adding a camera.
"""

import json
import logging

import httpx
import websockets

from ..config import Config
from ..wallfeed import WallConfig

log = logging.getLogger(__name__)


def camera_urls(cfg: Config, name: str) -> tuple[str, str]:
    host = cfg.ha.smartcam_host
    stream = cfg.go2rtc.restream(name, host=host)
    still = (f"http://{host}:{cfg.web.port}/api/wall.jpg" if name == "wall"   # the feed without starting its encoder
             else f"http://{host}:{cfg.web.port}/go2rtc/api/frame.jpeg?src={name}_sub")
    return stream, still


class HaCameras:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = httpx.AsyncClient(base_url=cfg.ha.url, timeout=60,
                                      headers={"Authorization": f"Bearer {cfg.ha.token}"})

    async def existing(self) -> dict[str, dict]:
        r = await self.http.get("/api/config/config_entries/entry", params={"domain": "generic"})
        r.raise_for_status()
        return {e["title"]: e for e in r.json()}

    async def _create(self, name: str) -> str:
        stream, still = camera_urls(self.cfg, name)
        base = "/api/config/config_entries/flow"
        flow = (await self.http.post(base, json={"handler": "generic", "show_advanced_options": True})).json()
        try:
            step = (await self.http.post(f"{base}/{flow['flow_id']}", json={
                "stream_source": stream, "still_image_url": still,
                "advanced": {"framerate": 2, "verify_ssl": False, "rtsp_transport": "tcp", "authentication": "basic"},
            })).json()
            if step.get("type") == "form" and step.get("step_id") == "user_confirm":
                step = (await self.http.post(f"{base}/{flow['flow_id']}", json={"confirmed_ok": True})).json()
            if step.get("type") != "create_entry":
                raise RuntimeError(f"Home Assistant did not accept the camera: {step.get('errors') or step}")
            return step["result"]["entry_id"]
        except Exception:
            await self.http.delete(f"{base}/{flow['flow_id']}")
            raise

    async def _name(self, entry_id: str, name: str) -> None:
        """Title the entry "SmartCam <name>" and its entity camera.smartcam_<name>."""
        url = self.cfg.ha.url.replace("http", "ws", 1).rstrip("/") + "/api/websocket"
        async with websockets.connect(url, max_size=None) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "auth", "access_token": self.cfg.ha.token}))
            await ws.recv()
            n = 0

            async def call(msg):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, **msg}))
                while True:
                    r = json.loads(await ws.recv())
                    if r.get("id") == n:
                        return r

            await call({"type": "config_entries/update", "entry_id": entry_id, "title": f"SmartCam {name}"})
            entities = (await call({"type": "config/entity_registry/list"}))["result"]
            for e in entities:
                if e.get("config_entry_id") == entry_id and e["entity_id"].startswith("camera."):
                    await call({"type": "config/entity_registry/update", "entity_id": e["entity_id"],
                                "name": f"SmartCam {name}", "new_entity_id": f"camera.smartcam_{name}"})

    async def ensure(self) -> dict[str, str]:
        """Create the missing cameras. Returns {camera: "created" | "exists" | error}."""
        if not self.cfg.go2rtc.rtsp_lan or not self.cfg.ha.smartcam_host:
            raise ValueError("needs go2rtc.rtsp_lan: true and ha.smartcam_host (the address Home Assistant reaches SmartCam on)")
        have = await self.existing()
        out = {}
        for name in [*self.cfg.cameras, *(["wall"] if WallConfig(**self.cfg.wall).enabled else [])]:
            if f"SmartCam {name}" in have:
                out[name] = "exists"
                continue
            try:
                await self._name(await self._create(name), name)
                out[name] = "created"
            except Exception as e:
                log.exception("Home Assistant camera %s", name)
                out[name] = str(e)
        return out
