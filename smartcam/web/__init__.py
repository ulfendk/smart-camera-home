"""FastAPI app: live view (via go2rtc's player), events with snapshots and clips, away mode.

All URLs in the UI are relative, so it also works behind a path prefix such as Home Assistant ingress.
go2rtc's player is proxied under go2rtc/ for the same reason: one port, reachable wherever the UI is.
"""

import asyncio
import re
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config
from ..db import Database
from ..events import EventEngine
from ..recorder import segment_start, segments
from ..settings import Settings

INDEX = (Path(__file__).parent / "index.html").read_text()

# Only the player and its stream endpoints; go2rtc's config API would expose camera credentials.
GO2RTC_FILES = {"stream.html", "video-stream.js", "video-rtc.js", "api/frame.jpeg"}


class Away(BaseModel):
    away: bool


class Review(BaseModel):
    review: str | None  # 'ok', 'false_positive' or null


class CollapseSlashes:
    """Treat //path as /path. Home Assistant ingress forwards the entry page as "//"."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and "//" in scope["path"]:
            scope = {**scope, "path": re.sub(r"/{2,}", "/", scope["path"])}
        await self.app(scope, receive, send)


def create_app(cfg: Config, db: Database, engine: EventEngine, restart: asyncio.Event | None = None,
               alarm=None, presence=None) -> FastAPI:
    app = FastAPI(title="SmartCam")
    app.add_middleware(CollapseSlashes)
    for sub in ("recordings", "snapshots", "clips"):
        (cfg.paths.data / sub).mkdir(parents=True, exist_ok=True)
        app.mount(f"/media/{sub}", StaticFiles(directory=cfg.paths.data / sub), name=sub)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX

    go2rtc = f"127.0.0.1:{cfg.go2rtc.api_port}"
    http = httpx.AsyncClient(base_url=f"http://{go2rtc}", timeout=10)

    @app.get("/go2rtc/{path:path}")
    async def go2rtc_file(path: str, request: Request) -> Response:
        if path not in GO2RTC_FILES:
            raise HTTPException(404)
        r = await http.get(f"/{path}", params=request.query_params)
        return Response(r.content, r.status_code, media_type=r.headers.get("content-type"))

    @app.websocket("/go2rtc/api/ws")
    async def go2rtc_ws(ws: WebSocket) -> None:
        """Relay the player's WebSocket (MSE video, WebRTC signalling) to go2rtc."""
        await ws.accept()
        try:
            async with websockets.connect(f"ws://{go2rtc}/api/ws?{ws.url.query}", max_size=None) as upstream:
                # Either side may go away at any moment (a page reload, a camera dropping out);
                # whichever relay ends first ends the session.
                async def down():
                    try:
                        async for msg in upstream:
                            await (ws.send_bytes(msg) if isinstance(msg, bytes) else ws.send_text(msg))
                    except Exception:
                        pass

                async def up():
                    try:
                        while True:
                            msg = await ws.receive()
                            if msg["type"] == "websocket.disconnect":
                                return
                            await upstream.send(msg.get("bytes") or msg.get("text"))
                    except Exception:
                        pass

                tasks = [asyncio.create_task(down()), asyncio.create_task(up())]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in tasks:
                    t.cancel()
        except (WebSocketDisconnect, websockets.ConnectionClosed, OSError):
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass  # already closed

    @app.get("/api/cameras")
    def cameras() -> list[dict]:
        return [{"name": c.name, "location": c.location, "zones": list(c.zones)} for c in cfg.cameras.values()]

    @app.get("/api/events")
    def events(limit: int = 100, camera: str | None = None, min_score: float = 0) -> list[dict]:
        return db.events(limit, camera, min_score)

    @app.post("/api/events/{event_id}/review")
    def review(event_id: int, body: Review) -> dict:
        if body.review not in (None, "ok", "false_positive"):
            raise HTTPException(400, "review must be ok, false_positive or null")
        if db.event(event_id) is None:
            raise HTTPException(404)
        db.update_event(event_id, review=body.review)
        return db.event(event_id)

    @app.get("/api/away")
    def get_away() -> Away:
        return Away(away=engine.away)

    @app.put("/api/away")
    async def put_away(body: Away) -> Away:
        await engine.set_away(body.away)
        return Away(away=engine.away)

    @app.get("/api/alarm")
    def get_alarm() -> dict:
        return {"state": alarm.state if alarm else "unavailable"}

    @app.post("/api/alarm/{command}")
    async def post_alarm(command: str) -> dict:
        """arm_away, disarm, or trigger (a test of the deterrence)."""
        if alarm is None:
            raise HTTPException(404, "alarm mode is off")
        if command == "trigger":
            await alarm.trigger("triggered from the web UI", delay=0)   # a test: no entry delay
        elif command in ("arm_away", "disarm"):
            await alarm.command(command)
        else:
            raise HTTPException(400, "command must be arm_away, disarm or trigger")
        return {"state": alarm.state}

    @app.post("/api/ha/cameras")
    async def ha_cameras() -> dict:
        """Create live camera entities in Home Assistant for every camera that has none yet."""
        if not cfg.ha.enabled:
            raise HTTPException(400, "the Home Assistant section is off")
        from ..ha.cameras import HaCameras
        try:
            return await HaCameras(cfg).ensure()
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/presence")
    async def presence_preview() -> dict:
        """Today's presence simulation plan (what would run while armed), and how much is learned."""
        if presence is None:
            raise HTTPException(404, "presence simulation is off")
        await presence.learn()
        return await presence.preview()

    @app.get("/api/config")
    def get_config() -> dict:
        """The config file as written (secrets masked), for the settings UI."""
        if cfg.path is None:
            raise HTTPException(404, "no config file")
        return Settings(cfg.path).read()

    @app.put("/api/config")
    def put_config(body: dict) -> dict:
        if cfg.path is None:
            raise HTTPException(404, "no config file")
        try:
            changed = Settings(cfg.path).write(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"changed": changed}

    @app.post("/api/restart")
    def post_restart() -> dict:
        """Restart smartcam to apply a saved config."""
        if restart is None:
            raise HTTPException(501, "restart not available")
        restart.set()
        return {"restarting": True}

    @app.get("/api/recordings/{camera}")
    def recordings(camera: str, start: float, end: float) -> list[dict]:
        return [{"start": segment_start(p), "url": f"media/recordings/{p.relative_to(cfg.paths.recordings)}"}
                for p in segments(cfg, camera, start, end)]

    return app
