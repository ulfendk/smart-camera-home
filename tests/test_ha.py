from pathlib import Path

from smartcam.config import load_config

TEST_CONFIG = Path(__file__).parent / "config.yml"
from smartcam.ha.mqtt import discovery_messages
from smartcam.ha.notify import alert_payload


def test_discovery_covers_away_and_every_camera():
    cfg = load_config(TEST_CONFIG)
    topics = [t for t, _ in discovery_messages(cfg)]
    assert f"{cfg.mqtt.discovery_prefix}/switch/smartcam/away/config" in topics
    assert len(topics) == 2 + 2 * len(cfg.cameras) + cfg.mqtt.alert_slots   # away, alarm, per camera, slots
    ids = [p["unique_id"] for _, p in discovery_messages(cfg)]
    assert len(ids) == len(set(ids))


def test_alert_payload_uses_its_own_alert_image_and_stable_tag():
    cfg = load_config(TEST_CONFIG)
    cfg.ha.event_url = "/smartcam/#event={id}"
    event = {"id": 7, "camera": "front_door", "label": "person", "reasons": ["person", "away", "night"]}
    p = alert_payload(cfg, event)
    assert p["data"]["entity_id"] == "camera.smartcam_alert_8"
    assert p["data"]["image"] == "/api/camera_proxy/camera.smartcam_alert_8"
    assert alert_payload(cfg, {**event, "id": 7 + cfg.mqtt.alert_slots})["data"]["entity_id"] == "camera.smartcam_alert_8"
    assert alert_payload(cfg, {**event, "id": 8})["data"]["entity_id"] == "camera.smartcam_alert_9"
    assert p["data"]["tag"] == "smartcam-7" and p["data"]["url"] == "/smartcam/#event=7"
    assert p["message"] == "away, night"
    assert alert_payload(cfg, event, clip_ready=True)["title"].startswith("Clip:")
