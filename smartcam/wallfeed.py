"""The wall as one feed: every camera tiled into a single video, for a TV, VLC or a weak tablet.

The tiles come from the frames detection already decodes (the newest per camera), so they are always
in step, a camera that is down shows "no signal" instead of stalling the rest, and people are outlined
just like in the browser's wall view. go2rtc runs the encoder on demand: its `wall` stream is an ffmpeg
that reads raw frames from a local socket here, so nothing is composed or encoded while nobody watches.
"""

import asyncio
import logging
import math
import time

import cv2
import numpy as np
from pydantic import BaseModel

from .config import Config

log = logging.getLogger(__name__)

RED = (55, 68, 219)      # BGR of the UI's error colour
STALE = 10.0             # seconds without a frame before a tile shows "no signal"


class WallConfig(BaseModel):
    enabled: bool = True
    cameras: list[str] = []      # default: all, in config order
    width: int = 1920
    fps: float = 5               # detection's frame rate; more only repeats frames
    bitrate: str = "4M"
    port: int = 8975             # local socket the encoder reads from


def layout(n: int, width: int) -> tuple[int, int, int, int]:
    """Columns, rows and the 16:9 tile size for n cameras across `width` pixels (even sizes for the encoder)."""
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    tw = width // cols // 2 * 2
    th = tw * 9 // 16 // 2 * 2
    return cols, rows, tw, th


def frame_size(cfg: WallConfig, n: int) -> tuple[int, int]:
    cols, rows, tw, th = layout(n, cfg.width)
    return cols * tw, rows * th


def compose(names: list[str], frames: dict, people: dict, cfg: WallConfig, now: float) -> np.ndarray:
    """The wall as one BGR image. `frames`: camera -> (ts, image); `people`: camera -> boxes (normalised)."""
    cols, rows, tw, th = layout(len(names), cfg.width)
    out = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, name in enumerate(names):
        x, y = (i % cols) * tw, (i // cols) * th
        tile = out[y:y + th, x:x + tw]
        ts, img = frames.get(name, (0.0, None))
        if img is None or now - ts > STALE:
            cv2.putText(tile, "no signal", (tw // 2 - 70, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2, cv2.LINE_AA)
        else:
            tile[:] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        boxes = people.get(name, [])
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(tile, (int(x1 * tw), int(y1 * th)), (int(x2 * tw), int(y2 * th)), RED, 2)
        if boxes:
            cv2.rectangle(tile, (1, 1), (tw - 2, th - 2), RED, 4)
        label = name + ("  person" if boxes else "")
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(tile, (8, th - lh - 22), (8 + lw + 12, th - 8), RED if boxes else (0, 0, 0), -1)
        cv2.putText(tile, label, (14, th - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    clock = time.strftime("%H:%M:%S", time.localtime(now))
    cv2.putText(out, clock, (out.shape[1] - 150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, clock, (out.shape[1] - 150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def encoder_source(cfg: Config) -> str:
    """go2rtc's source for the `wall` stream: ffmpeg encoding the raw frames served here."""
    wall = WallConfig(**cfg.wall)
    w, h = frame_size(wall, len(cameras(cfg)))
    codec = {
        "vaapi": "-vaapi_device /dev/dri/renderD128 -vf format=nv12,hwupload -c:v h264_vaapi",
        "videotoolbox": "-c:v h264_videotoolbox -realtime 1",
        "none": "-c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p",
    }[cfg.detect.hwaccel]
    return (f"exec:ffmpeg -hide_banner -loglevel error -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {wall.fps:g} "
            f"-i tcp://127.0.0.1:{wall.port} {codec} -b:v {wall.bitrate} -g {int(wall.fps * 2)} -bf 0 "
            f"-f rtsp {{output}}")


def cameras(cfg: Config) -> list[str]:
    wall = WallConfig(**cfg.wall)
    return [c for c in (wall.cameras or list(cfg.cameras)) if c in cfg.cameras]


class WallFeed:
    def __init__(self, cfg: Config, engine):
        self.cfg, self.engine = cfg, engine
        self.wall = WallConfig(**cfg.wall)
        self.names = cameras(cfg)

    def frame(self) -> np.ndarray:
        now = time.time()
        people = {c: boxes for c, (ts, boxes) in self.engine.people_boxes.items() if now - ts < 2}
        return compose(self.names, self.engine.live_frames, people, self.wall, now)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        log.info("wall feed: encoder connected")
        period, next_at = 1 / self.wall.fps, time.monotonic()
        try:
            while True:
                writer.write((await asyncio.to_thread(self.frame)).tobytes())
                await writer.drain()
                next_at += period
                await asyncio.sleep(max(0.0, next_at - time.monotonic()))
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            log.info("wall feed: encoder gone")

    async def run(self) -> None:
        server = await asyncio.start_server(self._serve, "127.0.0.1", self.wall.port)
        async with server:
            await server.serve_forever()
