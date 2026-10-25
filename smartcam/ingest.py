"""Decoded frames for detection, from a live go2rtc restream or from recorded files (replay)."""

import asyncio
import json
import logging
import subprocess
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

HWACCEL = {"videotoolbox": ["-hwaccel", "videotoolbox"], "vaapi": ["-hwaccel", "vaapi"], "none": []}


@dataclass
class Frame:
    camera: str
    ts: float            # wall-clock seconds
    image: np.ndarray    # BGR, HxWx3


class Sampler:
    """Thin a live frame stream to a target rate by arrival time, tolerating network jitter.

    A camera already at the target rate keeps every frame; a faster one keeps every n-th.
    """

    def __init__(self, fps: float):
        self.interval = 1 / fps
        self.due = 0.0

    def take(self, now: float) -> bool:
        if now < self.due - self.interval * 0.25:
            return False
        # Never schedule in the past, so frames buffered during a stall don't come out as a burst.
        self.due = max(self.due + self.interval, now + self.interval * 0.75)
        return True


def probe_size(url: str) -> tuple[int, int]:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", url],
        capture_output=True, text=True, timeout=20, check=True)
    s = json.loads(res.stdout)["streams"][0]
    return s["width"], s["height"]


def output_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if width <= max_width:
        return width, height
    # Keep it even, which ffmpeg's scaler requires.
    return max_width, round(height * max_width / width / 2) * 2


def decode_args(src: str, fps: float, size: tuple[int, int], hwaccel: str, live: bool) -> list[str]:
    # Live: decode every frame with minimal buffering and let live_frames() sample by wall clock;
    # ffmpeg's fps filter follows the camera's jittery timestamps and releases frames in bursts.
    live_args = ["-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "500000", "-analyzeduration", "1000000",
                 "-rtsp_transport", "tcp", "-timeout", "10000000"]
    vf = f"scale={size[0]}:{size[1]}" if live else f"fps={fps},scale={size[0]}:{size[1]}"
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        *HWACCEL[hwaccel], *(live_args if live else []),
        "-i", src, "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]


async def live_frames(cfg: Config, camera: str) -> AsyncIterator[Frame]:
    """Frames from the camera's detection restream; reconnects forever."""
    url = cfg.go2rtc.restream(f"{camera}_sub")
    while True:
        try:
            size = output_size(*await asyncio.to_thread(probe_size, url), cfg.detect.max_width)
            proc = await asyncio.create_subprocess_exec(
                *decode_args(url, cfg.detect.fps, size, cfg.detect.hwaccel, live=True),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            nbytes, sampler = size[0] * size[1] * 3, Sampler(cfg.detect.fps)
            try:
                while True:
                    buf = await proc.stdout.readexactly(nbytes)
                    now = time.time()
                    if sampler.take(now):
                        yield Frame(camera, now, np.frombuffer(buf, np.uint8).reshape(size[1], size[0], 3))
            finally:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
        except (asyncio.IncompleteReadError, subprocess.SubprocessError, KeyError, IndexError) as e:
            log.warning("%s: frame stream ended (%s); reconnecting in 5s", camera, type(e).__name__)
        await asyncio.sleep(5)


def file_frames(camera: str, files: list[tuple[float, Path]], fps: float, max_width: int,
                hwaccel: str = "none") -> Iterator[Frame]:
    """Frames from recorded segments, each given as (wall-clock start, path). Used by replay and tests."""
    for start, path in files:
        size = output_size(*probe_size(str(path)), max_width)
        proc = subprocess.Popen(decode_args(str(path), fps, size, hwaccel, live=False),
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        nbytes, i = size[0] * size[1] * 3, 0
        try:
            while len(buf := proc.stdout.read(nbytes)) == nbytes:
                yield Frame(camera, start + i / fps, np.frombuffer(buf, np.uint8).reshape(size[1], size[0], 3))
                i += 1
        finally:
            proc.kill()
            proc.wait()
