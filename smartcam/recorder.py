"""Continuous recording: one ffmpeg per camera copies the main stream into 10 s MP4 segments.

Layout: <data>/recordings/YYYY-MM-DD/HH/<camera>/MM.SS.mp4 (local time, the same layout as Frigate).
"""

import asyncio
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

from .config import CameraConfig, Config
from .supervise import supervise

log = logging.getLogger(__name__)


def ffmpeg_args(cfg: Config, cam: CameraConfig) -> list[str]:
    pattern = cfg.paths.recordings / "%Y-%m-%d" / "%H" / cam.name / "%M.%S.mp4"
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-rtsp_transport", "tcp", "-timeout", "10000000", "-i", cfg.go2rtc.restream(cam.name),
        # Video only: go2rtc's restream announces the camera audio track but sends no packets on it,
        # which makes ffmpeg abort the whole segment output.
        "-map", "0:v:0", "-c:v", "copy",
        # hvc1 tag so Safari/iOS can play the HEVC indoor streams.
        *(["-tag:v", "hvc1"] if cam.location == "indoor" else []),
        "-f", "segment", "-segment_time", str(cfg.record.segment_seconds), "-segment_format", "mp4",
        "-reset_timestamps", "1", "-strftime", "1", str(pattern),
    ]


def segment_start(path: Path) -> float:
    """Wall-clock start of a segment, parsed from its path."""
    hour_dir = path.parent.parent
    stamp = f"{hour_dir.parent.name} {hour_dir.name}:{path.stem.replace('.', ':')}"
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()


def segments(cfg: Config, camera: str, start: float, end: float) -> list[Path]:
    """Finished segments of a camera overlapping [start, end], oldest first."""
    found = []
    for path in cfg.paths.recordings.glob(f"*/*/{camera}/*.mp4"):
        s = segment_start(path)
        if s < end and s + cfg.record.segment_seconds > start:
            found.append((s, path))
    return [p for _, p in sorted(found)]


def ensure_dirs(cfg: Config, now: float | None = None) -> None:
    """Create this hour's and the next hour's segment directories; ffmpeg's segment muxer won't."""
    now = now or time.time()
    for t in (now, now + 3600):
        hour_dir = cfg.paths.recordings / datetime.fromtimestamp(t).strftime("%Y-%m-%d/%H")
        for name in cfg.cameras:
            (hour_dir / name).mkdir(parents=True, exist_ok=True)


def prune(cfg: Config, now: float | None = None) -> int:
    """Delete hour directories older than the retention period. Returns how many were removed."""
    cutoff = (now or time.time()) - cfg.record.retain_days * 86400
    removed = 0
    for hour_dir in cfg.paths.recordings.glob("*/*"):
        try:
            hour = datetime.strptime(f"{hour_dir.parent.name} {hour_dir.name}", "%Y-%m-%d %H").timestamp()
        except ValueError:
            continue
        if hour + 3600 < cutoff:
            shutil.rmtree(hour_dir)
            removed += 1
    for day_dir in cfg.paths.recordings.glob("*"):
        if day_dir.is_dir() and not any(day_dir.iterdir()):
            day_dir.rmdir()
    return removed


async def run(cfg: Config) -> None:
    ensure_dirs(cfg)
    tasks = [asyncio.create_task(supervise(f"record:{name}", lambda cam=cam: ffmpeg_args(cfg, cam)))
             for name, cam in cfg.cameras.items()]
    try:
        last_prune = 0.0
        while True:
            ensure_dirs(cfg)
            if time.monotonic() - last_prune > 600:
                last_prune = time.monotonic()
                if n := prune(cfg):
                    log.info("pruned %d hour(s) of recordings", n)
            await asyncio.sleep(60)
    finally:
        for t in tasks:
            t.cancel()
