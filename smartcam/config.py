"""Load the smartcam config, expanding ${VAR} from the environment / .env and {ip} per camera."""

import os
import re
import secrets
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).resolve().parent.parent
_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand_env(value: str) -> str:
    """Replace ${VAR} (required) and ${VAR:-default} from the environment."""
    def repl(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(f"environment variable {name} is not set (see .env.example)")

    return _ENV_RE.sub(repl, value)


def _polygon(v: list[float] | None) -> list[tuple[float, float]] | None:
    if v is None:
        return None
    if len(v) % 2:
        raise ValueError("polygon needs an even number of values")
    return [(float(v[i]), float(v[i + 1])) for i in range(0, len(v), 2)]


class CameraConfig(BaseModel):
    name: str
    location: Literal["indoor", "outdoor"]
    vendor: Literal["dahua", "generic"] = "generic"
    ip: str
    record: str
    detect: str
    zones: dict[str, list[tuple[float, float]]] = {}
    person_mask: list[tuple[float, float]] | None = None
    motion_mask: list[tuple[float, float]] | None = None
    power_switch: str | None = None   # HA switch powering the camera, cycled when its stream dies (Dahua: rebooted)
    speed: dict | None = None         # vehicle speed calibration, validated by speed.SpeedConfig

    @field_validator("zones", mode="before")
    @classmethod
    def _zones(cls, v):
        return {k: _polygon(p) for k, p in (v or {}).items()}

    @field_validator("person_mask", "motion_mask", mode="before")
    @classmethod
    def _masks(cls, v):
        return _polygon(v)


class MqttConfig(BaseModel):
    enabled: bool = False
    host: str
    port: int = 1883
    user: str | None = None
    password: str | None = None
    topic_prefix: str = "smartcam"
    discovery_prefix: str = "homeassistant"
    alert_slots: int = 20       # camera entities that hold one alert's image each, reused in turn


class HaConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    token: str = ""
    notify: list[str] = []          # notify services, e.g. mobile_app_my_phone
    event_url: str | None = None    # link opened from a notification; {id} is replaced by the event id
    smartcam_host: str | None = None   # this machine's address as Home Assistant sees it (for HA cameras)
    alarm_critical: bool = True     # alarm pushes as critical alerts: sound even in silent mode / Focus
    alarm_entity: str | None = None # follow this HA entity for arming, both ways (e.g. input_boolean.alarm)
    away_entity: str | None = None  # follow this HA entity for away mode, both ways


class PathsConfig(BaseModel):
    data: Path = Path("data")

    @property
    def recordings(self) -> Path:
        return self.data / "recordings"


class RecordConfig(BaseModel):
    segment_seconds: int = 10
    continuous_hours: float = 6   # every segment is kept this long; event clips are cut from them
    event_days: float = 10        # event clips and snapshots are kept this long
    min_free_gb: float = 10       # below this, the oldest recordings go first, then the oldest events


class DetectConfig(BaseModel):
    fps: float = 5
    max_width: int = 1280
    hwaccel: Literal["videotoolbox", "vaapi", "none"] = "none"
    model: Path = Path("models/yolo11m.onnx")


class Go2rtcConfig(BaseModel):
    binary: Path = Path("bin/go2rtc")
    api_port: int = 1984            # local only; the web UI proxies the player
    rtsp_port: int = 8554
    webrtc_port: int = 8555
    rtsp_lan: bool = False          # share the restreams on the LAN (e.g. for Home Assistant cameras)
    rtsp_user: str = "smartcam"
    rtsp_password: str | None = None   # generated and kept in the data dir when rtsp_lan is on

    def restream(self, stream: str, host: str = "127.0.0.1") -> str:
        auth = f"{self.rtsp_user}:{self.rtsp_password}@" if self.rtsp_password else ""
        return f"rtsp://{auth}{host}:{self.rtsp_port}/{stream}"


class WebConfig(BaseModel):
    port: int = 8971


class Config(BaseModel):
    cameras: dict[str, CameraConfig]
    mqtt: MqttConfig
    ha: HaConfig = HaConfig()
    paths: PathsConfig = PathsConfig()
    record: RecordConfig = RecordConfig()
    detect: DetectConfig = DetectConfig()
    go2rtc: Go2rtcConfig = Go2rtcConfig()
    web: WebConfig = WebConfig()
    rules: dict = {}  # validated by events.RulesConfig
    lights: dict = {}  # validated by lights.controller.LightsConfig
    watchdog: dict = {}  # validated by power.WatchdogConfig
    alarm: dict = {}  # validated by alarm.AlarmConfig
    presence: dict = {}  # validated by presence.PresenceConfig
    path: Path | None = None  # the file this was loaded from (the settings UI writes it)


def _secret(path: Path) -> str:
    """A generated secret, created once and kept (mode 600) so it survives restarts."""
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(18)
    path.write_text(value)
    path.chmod(0o600)
    return value


def _merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | None = None) -> Config:
    """Load the config: `path`, else $SMARTCAM_CONFIG, else config/smartcam.yml (copy it from
    config/smartcam.example.yml). An optional overlay in $SMARTCAM_OVERLAY is deep-merged over it."""
    path = path or Path(os.environ.get("SMARTCAM_CONFIG") or ROOT / "config" / "smartcam.yml")
    load_dotenv(path.parent / ".env")   # e.g. next to the config in a Docker data dir
    load_dotenv(ROOT / ".env")
    raw = yaml.safe_load(path.read_text())
    if overlay := os.environ.get("SMARTCAM_OVERLAY"):
        raw = _merge(raw, yaml.safe_load(Path(overlay).read_text()) or {})
    cameras = {}
    for name, cam in raw["cameras"].items():
        cam = {**cam, "name": name}
        for key in ("record", "detect"):
            cam[key] = _expand_env(cam[key]).replace("{ip}", cam["ip"])
        cameras[name] = cam
    mqtt = {k: _expand_env(v) if isinstance(v, str) else v for k, v in raw["mqtt"].items()}
    ha = {k: _expand_env(v) if isinstance(v, str) else v for k, v in raw.get("ha", {}).items()}
    cfg = Config(cameras=cameras, mqtt=mqtt, ha=ha, path=path,
                 **{k: raw[k] for k in ("paths", "record", "detect", "go2rtc", "web", "rules", "lights", "watchdog", "alarm", "presence") if k in raw})
    # Relative paths are relative to the repo root, not the working directory.
    if not cfg.paths.data.is_absolute():
        cfg.paths.data = ROOT / cfg.paths.data
    if not cfg.go2rtc.binary.is_absolute():
        cfg.go2rtc.binary = ROOT / cfg.go2rtc.binary
    if not cfg.detect.model.is_absolute():
        cfg.detect.model = ROOT / cfg.detect.model
    if cfg.go2rtc.rtsp_lan and not cfg.go2rtc.rtsp_password:
        cfg.go2rtc.rtsp_password = _secret(cfg.paths.data / "rtsp_password")
    return cfg
