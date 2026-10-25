import random

from smartcam.ingest import Sampler, output_size


def rate(fps_in: float, fps_out: float, jitter: float = 0.0, seconds: float = 60) -> float:
    rng, s = random.Random(1), Sampler(fps_out)
    times = [i / fps_in + rng.uniform(-jitter, jitter) for i in range(int(seconds * fps_in))]
    return sum(s.take(t) for t in sorted(times)) / seconds


def test_keeps_all_frames_at_target_rate_despite_jitter():
    assert rate(5, 5, jitter=0.03) > 4.9


def test_thins_faster_stream_to_target():
    assert 4.9 < rate(15, 5, jitter=0.01) < 5.1


def test_no_burst_after_stall():
    s = Sampler(5)
    assert s.take(0.0)
    # 20 frames buffered during a 4 s stall arrive within 100 ms.
    burst = [s.take(4.0 + i * 0.005) for i in range(20)]
    assert sum(burst) == 1


def test_output_size_keeps_aspect_and_even_height():
    assert output_size(2592, 1944, 1280) == (1280, 960)
    assert output_size(640, 360, 1280) == (640, 360)


def test_logged_ffmpeg_errors_hide_credentials():
    from smartcam.supervise import redact
    msg = "Error opening input file rtsp://smartcam:yo6Q-x_y@127.0.0.1:8554/entre. | rtsp://admin:p@ss@192.0.2.1/x"
    assert "yo6Q" not in redact(msg) and "p@ss" not in redact(msg)
    assert "rtsp://***@127.0.0.1:8554/entre" in redact(msg)


def test_privacy_cameras_record_only_while_away():
    import asyncio
    from pathlib import Path
    from smartcam import recorder
    from smartcam.config import load_config
    cfg = load_config(Path(__file__).parent / "config.yml")
    cfg.cameras["living_room"].record_at_home = False
    started = []

    async def fake_supervise(name, argv):
        started.append(name)
        await asyncio.sleep(3600)

    async def go():
        recorder.supervise, real = fake_supervise, recorder.supervise
        recorder.ensure_dirs, real_dirs = (lambda cfg, now=None: None), recorder.ensure_dirs
        try:
            task = asyncio.create_task(recorder.run(cfg, should_record=lambda n: False))
            await asyncio.sleep(0.05)
            task.cancel()
        finally:
            recorder.supervise, recorder.ensure_dirs = real, real_dirs
        assert sorted(started) == ["record:driveway", "record:front_door"]   # not the living room
    asyncio.run(go())
