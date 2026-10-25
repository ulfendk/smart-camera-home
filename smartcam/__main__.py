"""Run smartcam: go2rtc, recorders, live detection and the web UI.  Usage: uv run python -m smartcam"""

import asyncio
import logging
import os
import signal
import sys

import uvicorn

from . import detection, go2rtc, recorder, retention
from .config import load_config
from .db import Database
from .alarm import Alarm, AlarmConfig
from .events import EventEngine, RulesConfig
from .web import create_app


class Restart(Exception):
    """Raised to leave the task group cleanly and start again with the new config."""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    db = Database(cfg.paths.data / "smartcam.db")
    engine = EventEngine(cfg, db, RulesConfig(**cfg.rules))
    alarm = None
    if AlarmConfig(**cfg.alarm).enabled:
        alarm = Alarm(cfg, db, engine)
    if cfg.mqtt.enabled:
        from .ha.mqtt import HaMqtt
        mqtt = HaMqtt(cfg, engine, alarm)
    link = None
    if cfg.ha.enabled:
        from .ha.notify import HaNotifier
        HaNotifier(cfg, engine, alarm)
        if cfg.ha.alarm_entity or cfg.ha.away_entity:
            from .ha.link import HaLink
            link = HaLink(cfg, engine, alarm)
    presence = None
    from .presence import PresenceConfig, PresenceSimulator
    pc = PresenceConfig(**cfg.presence)
    if cfg.ha.enabled and alarm is not None and (pc.enabled or pc.lights):   # lights alone: learn and preview only
        presence = PresenceSimulator(cfg, db, alarm)
    watchdog = None
    from .power import PowerWatchdog
    if (wd := PowerWatchdog(cfg)).last_frame:   # any camera it can restart
        watchdog = wd
    restart = asyncio.Event()
    server = uvicorn.Server(uvicorn.Config(create_app(cfg, db, engine, restart, alarm, presence), host="0.0.0.0", port=cfg.web.port,
                                           log_level="warning", lifespan="off"))  # no startup hooks
    if alarm is not None:
        await alarm.startup()
    try:
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
            if link is not None:
                tg.create_task(link.run())
            if presence is not None:
                tg.create_task(presence.run())
            tg.create_task(_restart_on(restart))
    finally:
        if alarm is not None:
            await alarm.close()   # an alarm in progress switches its lights and sirens off


async def _restart_on(event: asyncio.Event) -> None:
    await event.wait()
    await asyncio.sleep(0.5)   # let the HTTP response go out
    logging.getLogger("smartcam").info("restarting to apply the new config")
    raise Restart


async def run_until_stopped() -> bool:
    """Run until stopped. Returns True if a restart was requested (to apply a new config)."""
    # `docker stop` sends SIGTERM: cancel cleanly so lights are switched back and recorders stop.
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    restart = False
    try:
        await main()
    except* Restart:
        restart = True
    return restart


if __name__ == "__main__":
    restart = False
    try:
        restart = asyncio.run(run_until_stopped())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    if restart:
        # Everything has shut down (lights back, recorders stopped); start over in this process.
        os.execv(sys.executable, [sys.executable, "-m", "smartcam"])
