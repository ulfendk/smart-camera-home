import asyncio

import numpy as np

from smartcam.detect import Detection
from smartcam.ingest import Frame
from smartcam.lights.controller import LightController, LightsConfig
from smartcam.lights.dahua import parse_table
from smartcam.motion import MotionResult
from smartcam.pipeline import FrameResult
from smartcam.tracker import Track, TrackerConfig


class FakeLight:
    def __init__(self):
        self.calls = []

    async def on(self):
        self.calls.append("on")

    async def off(self):
        self.calls.append("off")


def result(ts: float, luma: int, person: bool) -> FrameResult:
    frame = Frame("front_door", ts, np.full((90, 160, 3), luma, np.uint8))
    tracks = []
    if person:
        t = Track(1, ts - 2, TrackerConfig())
        for i in range(10):  # a walking person
            t.add(ts - 2 + i * 0.2, Detection("person", 0.8, (0.1 + i * 0.05, 0.3, 0.2 + i * 0.05, 0.8)), None)
        t.last_ts = ts
        tracks = [t]
    return FrameResult(frame, MotionResult(np.zeros((9, 16), np.uint8), 0), [], tracks)


def drive(ctrl, events):
    async def go():
        for ts, luma, person in events:
            await ctrl.handle(result(ts, luma, person), {1})
    asyncio.run(go())


def test_light_on_for_person_in_dark_then_off_after_timeout():
    light = FakeLight()
    ctrl = LightController(LightsConfig(on_seconds=60), light, "front_door")
    # Dark, person walks; the lit scene must not count as "not dark" and switch the light off.
    drive(ctrl, [(0, 20, True), (1, 200, True), (30, 200, False), (61, 200, False), (62, 200, False)])
    assert light.calls == ["on", "off"]


def test_no_light_in_daylight():
    light = FakeLight()
    drive(LightController(LightsConfig(), light, "front_door"), [(0, 150, True), (1, 150, True)])
    assert light.calls == []


def test_cooldown_prevents_flicker():
    light = FakeLight()
    ctrl = LightController(LightsConfig(on_seconds=5, cooldown=20), light, "front_door")
    drive(ctrl, [(0, 20, True), (6, 20, False), (10, 20, True), (27, 20, True)])
    assert light.calls == ["on", "off", "on"]


def test_parse_dahua_table():
    t = parse_table("table.Lighting_V2[0][1][0].Mode=Auto\r\ntable.Lighting_V2[0][1][0].MiddleLight[0].Light=50\r\n")
    assert t == {"Lighting_V2[0][1][0].Mode": "Auto", "Lighting_V2[0][1][0].MiddleLight[0].Light": "50"}
