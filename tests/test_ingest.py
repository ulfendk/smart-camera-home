import random

from smartcam.ingest import MAX_LAG, PTS, FrameClock, Sampler, output_size


def rate(fps_in: float, fps_out: float, jitter: float = 0.0, seconds: float = 60) -> float:
    rng, s = random.Random(1), Sampler(fps_out)
    times = [i / fps_in + rng.uniform(-jitter, jitter) for i in range(int(seconds * fps_in))]
    return sum(s.take(t) for t in sorted(times)) / seconds


def test_keeps_all_frames_at_target_rate_despite_jitter():
    assert rate(5, 5, jitter=0.03) > 4.9


def test_thins_faster_stream_to_target():
    assert 4.9 < rate(15, 5, jitter=0.01) < 5.1


def test_duplicate_capture_times_are_not_a_burst_of_frames():
    s = Sampler(5)
    assert s.take(0.0)
    # 20 frames with (almost) the same capture time: one of them.
    burst = [s.take(4.0 + i * 0.005) for i in range(20)]
    assert sum(burst) == 1


def bursty(seconds=30, fps=5):
    """A 5 fps camera whose frames arrive in bursts: captured every 200 ms, delivered up to 1.4 s late."""
    rng = random.Random(2)
    frames, deliver = [], 0.0
    for i in range(int(seconds * fps)):
        captured = 100 + i / fps
        if i % 7 == 0:
            deliver = captured + rng.uniform(0.6, 1.4)   # a stall, then everything since comes at once
        deliver = max(deliver, captured + 0.01)
        frames.append((i / fps + 0.033, deliver))        # (pts, arrival)
    return frames


def test_bursty_camera_keeps_its_frames_and_their_spacing():
    clock, s, frames = FrameClock(), Sampler(5), bursty()
    for pts, now in frames[:20]:                           # until a frame has come quickly, the delay is unknown
        s.take(clock.ts(pts, now))
    kept = [(t, now) for pts, now in frames[20:] if s.take(t := clock.ts(pts, now))]
    assert len(kept) == len(frames) - 20                   # sampling by arrival keeps about a third of them
    gaps = [b[0] - a[0] for a, b in zip(kept, kept[1:])]
    assert min(gaps) > 0.19 and max(gaps) < 0.21           # capture times, not arrival times
    assert all(t <= now for t, now in kept)


def test_frame_clock_follows_a_restarted_stream_and_never_runs_ahead():
    c = FrameClock()
    assert c.ts(10.0, 1000.0) == 1000.0
    assert abs(c.ts(10.2, 1000.9) - 1000.2) < 0.001        # late delivery: the capture time (plus a little creep)
    assert c.ts(0.0, 2000.0) == 2000.0                     # pts went back: a new stream
    assert c.ts(None, 2001.0) == 2001.0                    # no pts: arrival time
    assert c.ts(0.4, 2000.1) <= 2000.1


def test_frame_clock_recovers_when_the_stream_time_falls_behind():
    c = FrameClock()
    for i in range(50):                                    # 10 s at 5 fps, on time
        assert c.ts(i * 0.2, 1000 + i * 0.2) == 1000 + i * 0.2
    late = [c.ts(10 + i * 0.2, 1030 + i * 0.2) for i in range(75)]   # the stream's time is now 20 s behind
    assert all(1030 + i * 0.2 - t > MAX_LAG for i, t in enumerate(late[:50]))   # late, for a while
    assert all(t == 1030 + i * 0.2 for i, t in enumerate(late[-20:], 55))       # then on time again


def test_showinfo_line_gives_frame_number_and_pts():
    line = (b"[Parsed_showinfo_1 @ 0x5581] n:  42 pts: 378000 pts_time:4.2     duration:18000 fmt:bgr24 "
            b"sar:1/1 s:1280x720 i:P iskey:0 type:P checksum:00000000")
    m = PTS.search(line)
    assert (int(m[1]), float(m[2])) == (42, 4.2)


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
