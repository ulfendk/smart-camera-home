import asyncio
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from smartcam.config import load_config
from smartcam.db import Database
from smartcam.events import EventEngine, OpenEvent, RulesConfig
from smartcam.faces import FacesConfig, Recognition, align, distance, match, signature
from smartcam.people import People

TEST_CONFIG = Path(__file__).parent / "config.yml"
ROOT = Path(__file__).parent.parent


def unit(*v):
    v = np.array(v + (0,) * (512 - len(v)), np.float32)
    return v / np.linalg.norm(v)


def test_match_picks_the_best_person_above_the_threshold():
    gallery = {"Farmor": np.stack([unit(1, 0), unit(1, .2)]), "Regin": np.stack([unit(0, 1)])}
    assert match(unit(1, .1), gallery, 0.45)[0] == "Farmor"
    assert match(unit(0, 1, .2), gallery, 0.45)[0] == "Regin"
    assert match(unit(0, 0, 1), gallery, 0.45)[0] is None


def test_align_gives_an_arcface_sized_face():
    lm = np.array([[40, 50], [70, 50], [55, 70], [42, 90], [68, 90]], np.float32)
    assert align(np.zeros((200, 200, 3), np.uint8), lm).shape == (112, 112, 3)


def body(top, bottom):
    img = np.zeros((200, 80, 3), np.uint8)
    img[:100], img[100:] = top, bottom
    return img


def test_colours_tell_people_apart_but_not_the_light():
    red_blue, darker = body((30, 30, 200), (200, 60, 20)), body((20, 20, 140), (140, 40, 14))
    assert distance(signature(red_blue), signature(darker)) < 0.35
    assert distance(signature(red_blue), signature(body((40, 180, 40), (200, 60, 20)))) > 0.35


def test_people_from_home_assistant_are_family_and_stay():
    db = Database(Path(tempfile.mkdtemp()) / "t.db")
    people = People(db)
    people.sync_ha({"person.regin": "Regin"})
    people.add("Farmor", "family")
    people.add("Naboen", "neighbour")
    assert people.group("Regin") == "family" and people.group("Naboen") == "neighbour"
    with pytest.raises(ValueError):
        people.remove("Regin")
    people.add_face(7, unit(1, 0), None, 1.0)
    people.name_event(7, "Naboen")
    assert list(people.gallery()) == ["Naboen"]
    people.remove("Naboen")
    assert people.gallery() == {} and [p["name"] for p in people.all()] == ["Farmor", "Regin"]


class FakeModel:
    def __init__(self):
        self.faces = {}   # camera -> embedding to "find"

    def best_face(self, image):
        return None


def setup(recognize=("front_door",), faces=None):
    cfg = load_config(TEST_CONFIG)
    cfg.paths.data = Path(tempfile.mkdtemp())
    for c in recognize:
        cfg.cameras[c].recognize = True
    db = Database(cfg.paths.data / "t.db")
    engine = EventEngine(cfg, db, RulesConfig())
    people = People(db)
    stills = {}

    async def grab(cfg, http, camera):
        return stills.get(camera)
    model = FakeModel()
    rec = Recognition(cfg, engine, people, FacesConfig(enabled=True, interval=0, wait=0.3), model, grab)
    return engine, people, rec, stills, model, db


def open_event(engine, db, camera, start):
    ev_id = db.insert_event(camera=camera, label="person", start_ts=start, max_score=0.9)
    ev = OpenEvent(ev_id, camera, "outdoor", start, start)
    ev.score, ev.reasons = 1, ["person"]
    return ev


async def settle(rec):
    for _ in range(20):
        await asyncio.sleep(0.01)
        if not rec._tasks:
            return


def test_a_face_names_the_event_and_the_name_is_carried_to_the_next_camera():
    async def go():
        engine, people, rec, stills, model, db = setup()
        people.add("Farmor", "friend")
        people.add_face(0, unit(1, 0), None, 1.0, person="Farmor")
        model.best_face = lambda image: (unit(1, .1), np.zeros((50, 40, 3), np.uint8))
        frame = np.zeros((400, 400, 3), np.uint8)
        frame[:] = (30, 30, 200)
        stills["front_door"] = stills["driveway"] = frame
        now = time.time()
        door = open_event(engine, db, "front_door", now)
        rec.see(door, (0.2, 0.1, 0.6, 0.9), now)
        await settle(rec)
        assert door.person == "Farmor" and door.person_via is None and db.event(door.id)["person"] == "Farmor"
        drive = open_event(engine, db, "driveway", now + 20)          # no face there, same clothes
        rec.see(drive, (0.3, 0.1, 0.5, 0.9), now + 20)
        await settle(rec)
        assert drive.person == "Farmor" and drive.person_via == "front_door"
        assert any("probably Farmor" in r for r in db.event(drive.id)["reasons"])
    asyncio.run(go())


def test_no_name_is_carried_for_other_clothes_or_two_people_who_both_fit():
    async def go():
        engine, people, rec, stills, model, db = setup(recognize=())
        now = time.time()
        red, green = np.zeros((400, 400, 3), np.uint8), np.zeros((400, 400, 3), np.uint8)
        red[:], green[:] = (30, 30, 200), (40, 180, 40)
        stills["front_door"], stills["driveway"] = red, green
        a = open_event(engine, db, "front_door", now)
        await engine.set_person(a, "Farmor", "friend", 0.9, None)
        rec.see(a, (0.2, 0.1, 0.6, 0.9), now)
        b = open_event(engine, db, "driveway", now + 10)
        rec.see(b, (0.2, 0.1, 0.6, 0.9), now + 10)
        await settle(rec)
        assert b.person is None                                          # green is not red
        stills["living_room"] = red
        c = open_event(engine, db, "living_room", now + 15)
        d = open_event(engine, db, "living_room", now + 15)
        stills["driveway"] = red
        b2 = open_event(engine, db, "driveway", now + 16)
        await engine.set_person(c, "Naboen", "neighbour", 0.9, None)
        rec.see(c, (0.2, 0.1, 0.6, 0.9), now + 15)
        await settle(rec)
        rec.see(b2, (0.2, 0.1, 0.6, 0.9), now + 16)
        await settle(rec)
        assert b2.person is None                                         # Farmor and Naboen both fit
    asyncio.run(go())


def test_a_face_found_later_names_the_earlier_event_on_another_camera():
    async def go():
        engine, people, rec, stills, model, db = setup()
        people.add("Farmor", "friend")
        people.add_face(0, unit(1, 0), None, 1.0, person="Farmor")
        frame = np.zeros((400, 400, 3), np.uint8)
        frame[:] = (30, 30, 200)
        stills["front_door"] = stills["driveway"] = frame
        now = time.time()
        drive = open_event(engine, db, "driveway", now)               # seen first, from behind
        rec.see(drive, (0.3, 0.1, 0.5, 0.9), now)
        await settle(rec)
        assert drive.person is None
        model.best_face = lambda image: (unit(1, .1), np.zeros((50, 40, 3), np.uint8))
        door = open_event(engine, db, "front_door", now + 20)
        rec.see(door, (0.2, 0.1, 0.6, 0.9), now + 20)
        await settle(rec)
        assert door.person == "Farmor" and drive.person == "Farmor" and drive.person_via == "front_door"
    asyncio.run(go())


def test_family_working_from_home_is_not_pushed_but_a_friend_is():
    async def go():
        engine, people, rec, stills, model, db = setup()
        people.sync_ha({"person.regin": "Regin"})
        people.add("Farmor", "friend")
        people.add_face(0, unit(1, 0), None, 1.0, person="Regin")
        people.add_face(0, unit(0, 1), None, 1.0, person="Farmor")
        pushed = []
        engine.on_alert.append(lambda e: asyncio.sleep(0, pushed.append(e["id"])))
        await engine.set_wfh(True)
        stills["front_door"] = np.zeros((400, 400, 3), np.uint8)
        for who, vec in (("Regin", unit(1, .1)), ("Farmor", unit(.1, 1))):
            model.best_face = lambda image, vec=vec: (vec, np.zeros((50, 40, 3), np.uint8))
            ev = open_event(engine, db, "front_door", time.time() + 1)
            engine.open[("front_door", ev.id)] = ev
            rec.see(ev, (0.2, 0.1, 0.6, 0.9), ev.start)
            await engine._rescore(ev, ev.start)
            await asyncio.sleep(0.4)
            assert ev.person == who
        assert len(pushed) == 1 and db.event(pushed[0])["person"] == "Farmor"
    asyncio.run(go())


@pytest.mark.skipif(not (ROOT / "models/auraface.onnx").exists(), reason="face models not downloaded")
def test_real_models_embed_the_same_face_the_same():
    import cv2
    from smartcam.faces import FaceRecognizer
    fr = FaceRecognizer(FacesConfig(model=ROOT / "models/auraface.onnx", detector=ROOT / "models/yunet.onnx"), "CPU")
    img = np.full((240, 200, 3), 200, np.uint8)
    cv2.circle(img, (100, 120), 70, (150, 170, 210), -1)
    assert fr.best_face(img) is None or fr.best_face(img)[0].shape == (512,)   # a drawing need not be a face
