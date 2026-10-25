import asyncio
from pathlib import Path

from smartcam.config import load_config
from smartcam.ha.person_lights import PersonLights, PersonLightsConfig

TEST_CONFIG = Path(__file__).parent / "config.yml"


class FakeHa:
    def __init__(self, states):
        self.states, self.calls = states, []

    async def get(self, path):
        entity = path.rsplit("/", 1)[1]
        s = self.states.get(entity)
        return type("R", (), {"status_code": 200 if s else 404, "json": lambda self=None: s})()

    async def post(self, path, json):
        service, light = path.rsplit("/", 1)[1], json["entity_id"]
        self.calls.append((service, light, json.get("brightness_pct", json.get("brightness"))))
        if service == "turn_on":
            b = round(json["brightness_pct"] * 2.55) if "brightness_pct" in json else json.get("brightness")
            self.states[light] = {"state": "on", "attributes": {"brightness": b}}
        else:
            self.states[light] = {"state": "off", "attributes": {}}


def setup(sun="below_horizon"):
    cfg = load_config(TEST_CONFIG)
    cfg.cameras["front_door"].person_lights = ["light.porch", "light.shared"]
    cfg.cameras["driveway"].person_lights = ["light.shared"]
    ha = FakeHa({"sun.sun": {"state": sun, "attributes": {}},
                 "light.porch": {"state": "on", "attributes": {"brightness": 26}},
                 "light.shared": {"state": "off", "attributes": {}}})
    engine = type("E", (), {"on_open": [], "on_close": []})()
    return PersonLights(cfg, engine, PersonLightsConfig(seconds=0.05), http=ha), ha


def ev(cam):
    return {"camera": cam, "label": "person", "id": 1}


def test_full_after_dark_then_back_to_exactly_how_it_was():
    async def go():
        pl, ha = setup()
        await pl._person(ev("front_door"))
        assert ha.states["light.porch"]["attributes"]["brightness"] == 255 and ha.states["light.shared"]["state"] == "on"
        await pl._gone(ev("front_door"))
        await asyncio.sleep(0.1)
        assert ha.states["light.porch"] == {"state": "on", "attributes": {"brightness": 26}}   # dimmed again
        assert ha.states["light.shared"]["state"] == "off"                                     # off again
    asyncio.run(go())


def test_daylight_does_nothing():
    async def go():
        pl, ha = setup(sun="above_horizon")
        await pl._person(ev("front_door"))
        assert ha.calls == []
    asyncio.run(go())


def test_shared_light_waits_for_both_cameras_and_hand_changes_are_kept():
    async def go():
        pl, ha = setup()
        await pl._person(ev("front_door"))
        await pl._person(ev("driveway"))
        await pl._gone(ev("front_door"))
        await asyncio.sleep(0.1)
        assert ha.states["light.shared"]["state"] == "on"          # driveway still has a person
        ha.states["light.shared"] = {"state": "on", "attributes": {"brightness": 128}}   # someone dims it by hand
        await pl._gone(ev("driveway"))
        await asyncio.sleep(0.1)
        assert ha.states["light.shared"]["attributes"]["brightness"] == 128
    asyncio.run(go())
