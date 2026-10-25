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


def test_ha_camera_urls_use_the_password_protected_lan_restream():
    from smartcam.ha.cameras import camera_urls
    cfg = load_config(TEST_CONFIG)
    cfg.go2rtc.rtsp_lan, cfg.go2rtc.rtsp_password, cfg.ha.smartcam_host = True, "s3cret", "192.0.2.5"
    stream, still = camera_urls(cfg, "front_door")
    assert stream == "rtsp://smartcam:s3cret@192.0.2.5:8554/front_door"
    assert still == "http://192.0.2.5:8971/go2rtc/api/frame.jpeg?src=front_door_sub"


def test_go2rtc_api_is_local_and_lan_rtsp_has_a_password():
    from smartcam.go2rtc import go2rtc_config
    cfg = load_config(TEST_CONFIG)
    assert go2rtc_config(cfg)["api"]["listen"].startswith("127.0.0.1:")
    assert go2rtc_config(cfg)["rtsp"] == {"listen": "127.0.0.1:8554"}
    cfg.go2rtc.rtsp_lan, cfg.go2rtc.rtsp_password = True, "s3cret"
    assert go2rtc_config(cfg)["rtsp"] == {"listen": ":8554", "username": "smartcam", "password": "s3cret"}
    assert cfg.go2rtc.restream("x") == "rtsp://smartcam:s3cret@127.0.0.1:8554/x"


def test_detect_same_as_record_uses_one_camera_connection():
    from smartcam.go2rtc import go2rtc_config
    cfg = load_config(TEST_CONFIG)
    cam = cfg.cameras["living_room"]
    cam.detect = cam.record
    streams = go2rtc_config(cfg)["streams"]
    assert streams["living_room_sub"] == ["rtsp://127.0.0.1:8554/living_room"]
