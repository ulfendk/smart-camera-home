"""Cut event clips out of the continuous recording segments (stream copy, no re-encode)."""

import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .recorder import segment_start, segments


def frame_at(cfg: Config, camera: str, ts: float, out: Path) -> Path | None:
    """A full-resolution still from the recording at `ts` (the main stream), or None."""
    segs = segments(cfg, camera, ts, ts + 0.01)
    if not segs:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(ts - segment_start(segs[0]), 0):.2f}",
         "-i", str(segs[0]), "-frames:v", "1", "-q:v", "2", str(out)],
        check=True, capture_output=True, timeout=60)
    return out if out.exists() else None


def frames_between(cfg: Config, camera: str, start: float, end: float, fps: float = 5) -> Iterator[tuple[float, np.ndarray]]:
    """Full-resolution stills from the recording over [start, end], `fps` per second, one at a time.

    The timestamps follow the segment names, which are only accurate to a second or so.
    """
    segs = segments(cfg, camera, start, end)
    if not segs:
        return
    first = segment_start(segs[0])
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{s}'\n" for s in segs))
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-ss", f"{max(start - first, 0):.2f}", "-t", f"{end - start:.2f}", "-vf", f"fps={fps}",
             "-q:v", "2", f"{tmp}/%04d.jpg"],
            check=True, capture_output=True, timeout=120)
        for i, path in enumerate(sorted(Path(tmp).glob("*.jpg"))):
            yield max(start, first) + i / fps, cv2.imread(str(path))


def cut_clip(cfg: Config, camera: str, start: float, end: float, out: Path) -> Path | None:
    """Write [start, end] of a camera's recording to `out`. Returns None if no footage covers it.

    Stream copy can only start on a keyframe, so the clip may begin up to one GOP before `start`.
    """
    segs = segments(cfg, camera, start, end)
    if not segs:
        return None
    first = segment_start(segs[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for s in segs:
            f.write(f"file '{s}'\n")
        listing = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
             "-ss", f"{max(start - first, 0):.2f}", "-t", f"{end - start:.2f}",
             "-c", "copy", "-movflags", "+faststart", str(out)],
            check=True, capture_output=True, timeout=120)
    finally:
        Path(listing).unlink(missing_ok=True)
    return out
