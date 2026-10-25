"""Generate the go2rtc config from smartcam.yml and run go2rtc.

go2rtc is the only process that talks to the cameras. Everything else (recorder, detector, live view)
reads its restreams: <camera> is the main stream and <camera>_sub the detection stream.
"""

import yaml

from .config import Config
from .supervise import supervise


def go2rtc_config(cfg: Config) -> dict:
    streams = {}
    for name, cam in cfg.cameras.items():
        streams[name] = [cam.record]
        # detect == record: feed detection from the main restream, so the camera keeps one connection
        # (some cameras cannot serve two streams reliably). Detection scales the frames down anyway.
        streams[f"{name}_sub"] = [cfg.go2rtc.restream(name) if cam.detect == cam.record else cam.detect]
    g = cfg.go2rtc
    rtsp = {"listen": f"{'' if g.rtsp_lan else '127.0.0.1'}:{g.rtsp_port}"}
    if g.rtsp_password:
        rtsp |= {"username": g.rtsp_user, "password": g.rtsp_password}
    return {
        # The API (and its stream list, which holds the camera URLs) stays on this machine.
        "api": {"listen": f"127.0.0.1:{g.api_port}"},
        "rtsp": rtsp,
        "webrtc": {"listen": f":{cfg.go2rtc.webrtc_port}"},
        "log": {"level": "warn"},
        "streams": streams,
    }


async def run(cfg: Config) -> None:
    # The generated file contains camera credentials, so it lives in the gitignored data dir.
    path = cfg.paths.data / "go2rtc.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(go2rtc_config(cfg), sort_keys=False))
    path.chmod(0o600)
    await supervise("go2rtc", lambda: [str(cfg.go2rtc.binary), "-config", str(path)])
