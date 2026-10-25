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
