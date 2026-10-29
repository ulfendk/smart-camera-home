"""Full-resolution stills from go2rtc (the cameras' main streams), and joining several into one picture."""

import cv2
import httpx
import numpy as np

from .config import Config


async def grab(cfg: Config, http: httpx.AsyncClient, camera: str) -> np.ndarray | None:
    """The camera's current main-stream picture (BGR), or None when go2rtc has none."""
    try:
        r = await http.get(f"http://127.0.0.1:{cfg.go2rtc.api_port}/api/frame.jpeg", params={"src": camera})
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def join(images: list[np.ndarray], height: int) -> np.ndarray:
    """Side by side at the same height, with a thin black gap (iOS notifications carry one picture)."""
    scaled = [cv2.resize(i, (round(i.shape[1] * height / i.shape[0]), height), interpolation=cv2.INTER_AREA) for i in images]
    gap = np.zeros((height, 6, 3), np.uint8)
    out = scaled[0]
    for img in scaled[1:]:
        out = np.hstack([out, gap, img])
    return out
