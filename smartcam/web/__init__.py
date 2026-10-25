"""FastAPI app: live view (via go2rtc's player), events with snapshots and clips, away mode."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config
from ..db import Database
from ..events import EventEngine
from ..recorder import segment_start, segments

INDEX = (Path(__file__).parent / "index.html").read_text()


class Away(BaseModel):
    away: bool


class Review(BaseModel):
    review: str | None  # 'ok', 'false_positive' or null


def create_app(cfg: Config, db: Database, engine: EventEngine) -> FastAPI:
    app = FastAPI(title="smartcam")
    for sub in ("recordings", "snapshots", "clips"):
        (cfg.paths.data / sub).mkdir(parents=True, exist_ok=True)
        app.mount(f"/media/{sub}", StaticFiles(directory=cfg.paths.data / sub), name=sub)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> str:
        return INDEX.replace("{{GO2RTC}}", f"http://{request.url.hostname}:{cfg.go2rtc.api_port}")

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

    @app.get("/api/recordings/{camera}")
    def recordings(camera: str, start: float, end: float) -> list[dict]:
        return [{"start": segment_start(p), "url": f"/media/recordings/{p.relative_to(cfg.paths.recordings)}"}
                for p in segments(cfg, camera, start, end)]

    return app
