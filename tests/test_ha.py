from pathlib import Path

from smartcam.config import load_config

TEST_CONFIG = Path(__file__).parent / "config.yml"
from smartcam.ha.mqtt import discovery_messages
from smartcam.ha.notify import alert_payload


def test_discovery_covers_away_and_every_camera():
    cfg = load_config(TEST_CONFIG)
    topics = [t for t, _ in discovery_messages(cfg)]
    assert f"{cfg.mqtt.discovery_prefix}/switch/smartcam/away/config" in topics
    assert f"{cfg.mqtt.discovery_prefix}/switch/smartcam/wfh/config" in topics
    assert len(topics) == 3 + 3 * len(cfg.cameras) + cfg.mqtt.alert_slots   # away, wfh, alarm, per camera, slots
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


def test_alarm_push_is_time_sensitive_or_critical_with_its_own_image():
    from smartcam.ha.notify import alarm_payload
    cfg = load_config(TEST_CONFIG)
    cfg.ha.alarm_critical = False
    p = alarm_payload(cfg, "living_room: indoor while away", {"id": 7})
    assert p["data"]["push"]["interruption-level"] == "time-sensitive"
    assert p["data"]["priority"] == "high" and p["data"]["entity_id"] == "camera.smartcam_alert_8"
    cfg.ha.alarm_critical = True
    p = alarm_payload(cfg, "test", None)
    assert p["data"]["push"]["interruption-level"] == "critical" and p["data"]["push"]["sound"]["critical"] == 1
    assert "entity_id" not in p["data"]


def test_alarm_push_goes_to_alarm_notify_and_alerts_to_notify():
    import asyncio
    from smartcam.ha.notify import HaNotifier
    cfg = load_config(TEST_CONFIG)
    cfg.ha.notify, cfg.ha.alarm_notify = ["mobile_app_me"], ["household"]
    sent = []

    class Http:
        async def post(self, url, json):
            sent.append(url.rsplit("/", 1)[1])
            return type("R", (), {"raise_for_status": lambda self: None})()

    class Engine:
        on_alert, on_clip, on_speed = [], [], []

    n = HaNotifier(cfg, Engine())
    n.http = Http()
    asyncio.run(n.alarm("indoor while away", None))
    asyncio.run(n.alert({"id": 1, "camera": "front_door", "label": "person", "reasons": ["person"]}))
    assert sent == ["household", "mobile_app_me"]
    cfg.ha.alarm_notify = []
    asyncio.run(n.alarm("test", None))
    assert sent[-1] == "mobile_app_me"
