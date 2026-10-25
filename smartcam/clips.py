"""Cut event clips out of the continuous recording segments (stream copy, no re-encode)."""

import subprocess
import tempfile
from pathlib import Path

from .config import Config
from .recorder import segment_start, segments


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
