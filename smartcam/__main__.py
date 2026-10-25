"""Run smartcam: go2rtc, recorders, live detection and the web UI.  Usage: uv run python -m smartcam"""

import asyncio
import logging
import signal

import uvicorn

from . import detection, go2rtc, recorder, retention
from .config import load_config
from .db import Database
from .events import EventEngine, RulesConfig
from .web import create_app


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    db = Database(cfg.paths.data / "smartcam.db")
    engine = EventEngine(cfg, db, RulesConfig(**cfg.rules))
    if cfg.mqtt.enabled:
        from .ha.mqtt import HaMqtt
        mqtt = HaMqtt(cfg, engine)
    if cfg.ha.enabled:
        from .ha.notify import HaNotifier
        HaNotifier(cfg, engine)
    watchdog = None
    from .power import PowerWatchdog
    if (wd := PowerWatchdog(cfg)).last_frame:   # any camera it can restart
        watchdog = wd
    server = uvicorn.Server(uvicorn.Config(create_app(cfg, db, engine), host="0.0.0.0", port=cfg.web.port,
                                           log_level="warning", lifespan="off"))  # no startup hooks
    async with asyncio.TaskGroup() as tg:
        tg.create_task(go2rtc.run(cfg))
        await asyncio.sleep(2)  # let go2rtc open its RTSP port before anything connects
        tg.create_task(recorder.run(cfg))
        tg.create_task(retention.run(cfg, db))
        tg.create_task(detection.run(cfg, engine, watchdog))
        if watchdog is not None:
            tg.create_task(watchdog.run())
        tg.create_task(server.serve())
        if cfg.mqtt.enabled:
            tg.create_task(mqtt.run())


async def run_until_stopped() -> None:
    # `docker stop` sends SIGTERM: cancel cleanly so lights are switched back and recorders stop.
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    await main()


if __name__ == "__main__":
    try:
        asyncio.run(run_until_stopped())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
