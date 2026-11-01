"""Decoded frames for detection, from a live go2rtc restream or from recorded files (replay)."""

import asyncio
import itertools
import json
import logging
import re
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


MAX_LAG = 3.0   # seconds: a frame captured longer ago than this is too late to be worth detecting
PTS = re.compile(rb"\bn:\s*(\d+)\b.*?\bpts_time:\s*(-?[\d.]+)")   # showinfo: frame number, pts


class FrameClock:
    """When each frame was captured, from the stream's own timestamps (pts), on the wall clock.

    Some cameras deliver frames in bursts: several within milliseconds, then nothing for a second. Arrival
    time would give a burst the same timestamp, which throws off tracking and speeds. The pts keep the real
    spacing; the offset to the wall clock is the smallest seen (the frame that came quickest), allowed to
    creep up slowly for a camera clock that runs slow, and reset when the stream restarts.
    """

    CREEP = 0.002   # s of offset per s of stream: far more than any clock drift, far less than a burst

    def __init__(self):
        self.offset: float | None = None
        self.last_pts = 0.0

    def ts(self, pts: float | None, now: float) -> float:
        if pts is None:
            return now
        off = now - pts
        if self.offset is None or pts < self.last_pts or pts - self.last_pts > 30:
            self.offset = off   # first frame, or the stream restarted / jumped
        else:
            self.offset = min(off, self.offset + self.CREEP * (pts - self.last_pts))
        self.last_pts = pts
        return min(self.offset + pts, now)


class Sampler:
    """Thin a live frame stream to a target rate by capture time, tolerating jitter.

    A camera at the target rate keeps every frame, also when its frames arrive in bursts (their capture
    times are still spaced out); a faster one keeps every n-th.
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
    # showinfo prints each live frame's pts on stderr (at info level), for FrameClock.
    live_args = ["-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "500000", "-analyzeduration", "1000000",
                 "-rtsp_transport", "tcp", "-timeout", "10000000"]
    vf = f"scale={size[0]}:{size[1]},showinfo" if live else f"fps={fps},scale={size[0]}:{size[1]}"
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "info" if live else "error", "-nostats", "-nostdin",
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
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            nbytes, sampler, clock = size[0] * size[1] * 3, Sampler(cfg.detect.fps), FrameClock()
            pts: dict[int, float] = {}
            reader = asyncio.create_task(_read_pts(proc.stderr, pts))
            try:
                for n in itertools.count():
                    buf = await proc.stdout.readexactly(nbytes)
                    now = time.time()
                    # showinfo logs a frame before ffmpeg writes it, but the two pipes are read separately:
                    # give stderr a moment, unless it has never carried a pts (then arrival time it is).
                    for _ in range(5 if pts or clock.offset is not None else 0):
                        if n in pts:
                            break
                        await asyncio.sleep(0.002)
                    ts = clock.ts(pts.pop(n, None), now)
                    for old in [k for k in pts if k < n]:
                        del pts[old]
                    if now - ts <= MAX_LAG and sampler.take(ts):
                        yield Frame(camera, ts, np.frombuffer(buf, np.uint8).reshape(size[1], size[0], 3))
            finally:
                reader.cancel()
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
        except (asyncio.IncompleteReadError, subprocess.SubprocessError, KeyError, IndexError) as e:
            log.warning("%s: frame stream ended (%s); reconnecting in 5s", camera, type(e).__name__)
        await asyncio.sleep(5)


async def _read_pts(stderr: asyncio.StreamReader, out: dict[int, float]) -> None:
    async for line in stderr:
        if m := PTS.search(line):
            out[int(m[1])] = float(m[2])


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
