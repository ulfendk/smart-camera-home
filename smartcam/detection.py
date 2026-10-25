"""Live detection: one pipeline per camera fed from its go2rtc sub-stream, results into the event engine."""

import asyncio
import logging
import time
from urllib.parse import urlsplit

from .config import CameraConfig, Config
from .events import EventEngine
from .ingest import Frame, live_frames
from .lights.controller import LightController, LightsConfig
from .lights.dahua import DahuaLight
from .pipeline import CameraPipeline
from .power import PowerWatchdog
from .speed import SpeedConfig, SpeedMeter
from .static_map import StaticMap

log = logging.getLogger(__name__)


async def _latest(cfg: Config, camera: str, slot: asyncio.Queue) -> None:
    """Keep only the newest frame in `slot`, so a slow consumer processes fresh frames, not a backlog."""
    async for frame in live_frames(cfg, camera):
        if slot.full():
            slot.get_nowait()
        slot.put_nowait(frame)


async def make_light(cfg: Config, cam: CameraConfig) -> LightController | None:
    lights = LightsConfig(**cfg.lights)
    wanted = lights.cameras or [c.name for c in cfg.cameras.values() if c.vendor == "dahua" and c.location == "outdoor"]
    if not lights.enabled or cam.name not in wanted:
        return None
    url = urlsplit(cam.record)
    light = DahuaLight(cam.name, cam.ip, url.username, url.password, cfg.paths.data, dry_run=lights.dry_run)
    try:
        await light.restore_after_crash()
    except Exception:
        log.exception("light %s: could not restore settings", cam.name)
    return LightController(lights, light, cam.name)


async def run_camera(cfg: Config, cam: CameraConfig, detector, engine: EventEngine,
                     watchdog: PowerWatchdog | None = None) -> None:
    static_map = StaticMap(cfg.paths.data / "static_map" / f"{cam.name}.npz")
    pipeline = CameraPipeline(cam, detector, static_map)
    slot: asyncio.Queue[Frame] = asyncio.Queue(maxsize=1)
    reader = asyncio.create_task(_latest(cfg, cam.name, slot))
    light = await make_light(cfg, cam)
    speed = SpeedConfig(**(cam.speed or {}))
    meter = SpeedMeter(cam.name, speed) if speed.ready else None
    last_save = 0.0
    stats = {"frames": 0, "detects": 0, "busy": 0.0, "since": time.monotonic()}
    try:
        while True:
            frame = await slot.get()
            if watchdog is not None:
                watchdog.seen(cam.name)
            t0 = time.monotonic()
            res = await asyncio.to_thread(pipeline.process, frame)
            stats["busy"] += time.monotonic() - t0
            stats["frames"] += 1
            stats["detects"] += res.detections is not None
            if (elapsed := time.monotonic() - stats["since"]) >= 300:
                log.info("%s: %.1f fps processed, %d%% detected, %.0f ms/frame, %d tracks",
                         cam.name, stats["frames"] / elapsed, 100 * stats["detects"] / max(stats["frames"], 1),
                         1000 * stats["busy"] / max(stats["frames"], 1), len(res.tracks))
                stats = {"frames": 0, "detects": 0, "busy": 0.0, "since": time.monotonic()}
            await engine.handle(res)
            if meter is not None:
                for reading in meter.update(res.tracks):
                    await engine.record_speed(reading, frame)
            if light is not None:
                await light.handle(res, pipeline.confirmed_ids)
            if frame.ts - last_save > 600:
                last_save = frame.ts
                static_map.save()
    finally:
        reader.cancel()
        static_map.save()
        if light is not None:
            await light.light.off()


async def run(cfg: Config, engine: EventEngine, watchdog: PowerWatchdog | None = None) -> None:
    from .detect.yolo import YoloDetector  # imported here so the recorder-only tools don't load onnxruntime

    detector = await asyncio.to_thread(YoloDetector, cfg.detect.model)
    async with asyncio.TaskGroup() as tg:
        for cam in cfg.cameras.values():
            tg.create_task(run_camera(cfg, cam, detector, engine, watchdog))
