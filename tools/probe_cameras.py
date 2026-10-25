"""Read-only probe of every camera: stream parameters, a snapshot, and Dahua capabilities.

Usage: uv run python tools/probe_cameras.py [camera ...]
Writes snapshots and a JSON report to data/probe/. Never changes camera settings.
"""

import json
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartcam.config import ROOT, load_config  # noqa: E402

OUT = ROOT / "data" / "probe"

# Dahua config tables that tell us how the white light / day-night switching is driven.
DAHUA_QUERIES = {
    "device_type": "magicBox.cgi?action=getDeviceType",
    "software": "magicBox.cgi?action=getSoftwareVersion",
    "lighting_v2": "configManager.cgi?action=getConfig&name=Lighting_V2",
    "lighting": "configManager.cgi?action=getConfig&name=Lighting",
    "day_night": "configManager.cgi?action=getConfig&name=VideoInDayNight",
    "video_in_options": "configManager.cgi?action=getConfig&name=VideoInOptions",
    "alarm_lighting": "configManager.cgi?action=getConfig&name=AlarmLighting",
}


def redact(url: str) -> str:
    scheme, rest = url.split("://", 1)
    return f"{scheme}://***@{rest.split('@', 1)[-1]}"


def ffprobe(url: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-timeout", "5000000",
           "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,profile,width,height,avg_frame_rate,r_frame_rate",
           "-of", "json", url]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if res.returncode:
        return {"error": res.stderr.strip().replace(url, redact(url))[-300:]}
    streams = json.loads(res.stdout).get("streams", [])
    return streams[0] if streams else {"error": "no video stream"}


def snapshot(url: str, dest: Path) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y", "-rtsp_transport", "tcp", "-timeout", "5000000",
           "-i", url, "-frames:v", "1", str(dest)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def dahua(ip: str, user: str, password: str) -> dict:
    out = {}
    with httpx.Client(auth=httpx.DigestAuth(user, password), timeout=5) as client:
        for key, query in DAHUA_QUERIES.items():
            try:
                r = client.get(f"http://{ip}/cgi-bin/{query}")
                out[key] = r.text.strip() if r.status_code == 200 else f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                out[key] = f"error: {e}"
    return out


def main() -> None:
    cfg = load_config()
    names = sys.argv[1:] or list(cfg.cameras)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in names:
        cam = cfg.cameras[name]
        print(f"== {name} ({cam.ip})", flush=True)
        entry = {"record": ffprobe(cam.record), "detect": ffprobe(cam.detect)}
        for role in ("record", "detect"):
            s = entry[role]
            print(f"  {role:6}: " + (s["error"] if "error" in s else
                  f"{s['codec_name']} {s['width']}x{s['height']} @ {s['avg_frame_rate']}"))
        entry["snapshot"] = snapshot(cam.detect, OUT / f"{name}.jpg")
        if cam.vendor == "dahua":
            user, password = cam.record.split("://", 1)[1].split("@", 1)[0].split(":", 1)
            entry["dahua"] = dahua(cam.ip, user, password)
            print(f"  dahua : {entry['dahua']['device_type']} | {entry['dahua']['software']}")
        report[name] = entry
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
