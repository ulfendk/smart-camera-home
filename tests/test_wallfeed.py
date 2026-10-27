import asyncio

import numpy as np

from smartcam.wallfeed import WallConfig, WallFeed, compose, frame_size, layout

RED = (55, 68, 219)


def test_layout_fits_cameras_in_a_near_square_grid():
    assert layout(7, 1920) == (3, 3, 640, 360)
    assert layout(4, 1920) == (2, 2, 960, 540)
    assert layout(1, 1920) == (1, 1, 1920, 1080)
    assert frame_size(WallConfig(), 7) == (1920, 1080)


def test_compose_tiles_outlines_people_and_marks_dead_cameras():
    grey = np.full((720, 1280, 3), 128, np.uint8)
    frames = {"a": (100.0, grey), "b": (100.0, grey), "c": (50.0, grey)}     # c last seen 50 s ago
    img = compose(["a", "b", "c", "d"], frames, {"b": [(0.4, 0.4, 0.6, 0.9)]}, WallConfig(width=1280), now=100.0)
    assert img.shape == (720, 1280, 3)
    assert tuple(img[180, 320]) == (128, 128, 128)                 # a: the camera picture
    assert tuple(img[2, 960]) == RED                               # b: red outline around a tile with a person
    assert tuple(img[380, 20]) == (0, 0, 0)                        # c: stale, "no signal" on black
    assert tuple(img[380, 660]) == (0, 0, 0)                       # d: never seen


def test_feed_serves_raw_frames_to_the_encoder(tmp_path):
    from types import SimpleNamespace
    cfg = SimpleNamespace(cameras={"a": None, "b": None}, wall={"width": 640, "fps": 20, "port": 0})
    import time
    engine = SimpleNamespace(people_boxes={}, live_frames={"a": (time.time(), np.zeros((720, 1280, 3), np.uint8))})
    feed = WallFeed(cfg, engine)
    size = 640 * 180 * 3          # two cameras: 2 x 1 tiles of 320 x 180

    async def main():
        server = await asyncio.start_server(feed._serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        data = await reader.readexactly(size * 2)
        writer.close()
        server.close()
        return data

    assert len(asyncio.run(main())) == size * 2


def test_go2rtc_gets_an_on_demand_wall_stream():
    from pathlib import Path
    from smartcam.config import load_config
    from smartcam.go2rtc import go2rtc_config
    cfg = load_config(Path(__file__).parent / "config.yml")
    src = go2rtc_config(cfg)["streams"]["wall"][0]
    assert src.startswith("exec:ffmpeg") and "tcp://127.0.0.1:8975" in src and src.endswith("-f rtsp {output}")
    cfg.wall = {"enabled": False}
    assert "wall" not in go2rtc_config(cfg)["streams"]
