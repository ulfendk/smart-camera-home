"""Keep the cameras' clocks right: the time in their picture, their own logs and events depend on it.

Dahua cameras: the clock is compared with this machine's and set when it is more than `max_drift`
seconds off (works whether or not the camera can reach an NTP server). ONVIF cameras (with
`onvif_port` set): the time zone is set to this machine's, with its daylight-saving rules; their own
NTP keeps the time.
"""

import asyncio
import base64
import datetime as dt
import hashlib
import logging
import os
import zoneinfo
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from .config import CameraConfig, Config

log = logging.getLogger(__name__)


class TimeSyncConfig(BaseModel):
    enabled: bool = True
    interval_hours: float = 6
    max_drift: float = 2            # seconds


def posix_tz(name: str) -> str | None:
    """The POSIX TZ string (e.g. CET-1CEST,M3.5.0,M10.5.0/3) from the zone file's footer."""
    for root in zoneinfo.TZPATH:
        p = Path(root) / name
        if p.is_file():
            lines = p.read_bytes().rstrip(b"\n").split(b"\n")
            return lines[-1].decode() if lines and lines[-1] else None
    return None


def local_zone() -> str | None:
    if tz := os.environ.get("TZ"):
        return tz
    try:
        return str(Path("/etc/localtime").resolve()).split("zoneinfo/")[1]
    except (IndexError, OSError):
        return None


async def sync_dahua(cam: CameraConfig, max_drift: float) -> float | None:
    """Returns the drift corrected (seconds), or None if it was within bounds."""
    u = urlsplit(cam.record)
    async with httpx.AsyncClient(auth=httpx.DigestAuth(u.username, u.password), timeout=8) as http:
        r = await http.get(f"http://{cam.ip}/cgi-bin/global.cgi?action=getCurrentTime")
        camera = dt.datetime.strptime(r.text.strip().split("=", 1)[1], "%Y-%m-%d %H:%M:%S")
        drift = (camera - dt.datetime.now()).total_seconds()
        if abs(drift) <= max_drift:
            return None
        now = dt.datetime.now().strftime("%Y-%m-%d%%20%H:%M:%S")
        r = await http.get(f"http://{cam.ip}/cgi-bin/global.cgi?action=setCurrentTime&time={now}")
        if "OK" not in r.text:
            raise RuntimeError(r.text.strip()[:100])
        return drift


def _wsse(user: str, password: str) -> str:
    nonce = os.urandom(16)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    ns = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-"
    return (f'<s:Header><Security s:mustUnderstand="1" xmlns="{ns}wssecurity-secext-1.0.xsd"><UsernameToken>'
            f"<Username>{user}</Username>"
            f'<Password Type="{ns}username-token-profile-1.0#PasswordDigest">{digest}</Password>'
            f'<Nonce EncodingType="{ns}soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>'
            f'<Created xmlns="{ns}wssecurity-utility-1.0.xsd">{created}</Created></UsernameToken></Security></s:Header>')


async def sync_onvif_zone(cam: CameraConfig, tz: str) -> bool | None:
    """Set the camera's time zone (keeping its NTP). True if it changed, False if it was right already,
    None if the camera accepted the change but ignores it (some firmware does)."""
    url = f"http://{cam.ip}:{cam.onvif_port}/onvif/device_service"
    head = {"Content-Type": "application/soap+xml"}
    env = '<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">{}<s:Body>{}</s:Body></s:Envelope>'
    async with httpx.AsyncClient(timeout=8) as http:
        r = await http.post(url, headers=head, content=env.format(
            "", '<GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/>'))
        if f"<tt:TZ>{tz}</tt:TZ>" in r.text or f">{tz}<" in r.text:
            return False
        ntp = "NTP" if ">NTP<" in r.text else "Manual"
        u = urlsplit(cam.record)
        r = await http.post(url, headers=head, content=env.format(_wsse(u.username, u.password), (
            '<SetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">'
            f"<DateTimeType>{ntp}</DateTimeType><DaylightSavings>false</DaylightSavings>"
            f"<TimeZone><tt:TZ>{tz}</tt:TZ></TimeZone></SetSystemDateAndTime>")))
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        r = await http.post(url, headers=head, content=env.format(
            "", '<GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/>'))
        return True if f">{tz}<" in r.text else None


async def run(cfg: Config) -> None:
    tsc = TimeSyncConfig(**cfg.timesync)
    if not tsc.enabled:
        return
    zone = local_zone()
    tz = posix_tz(zone) if zone else None
    ignored: set[str] = set()   # cameras that accept a time zone over ONVIF but ignore it
    while True:
        for name, cam in cfg.cameras.items():
            try:
                if cam.vendor == "dahua":
                    if (drift := await sync_dahua(cam, tsc.max_drift)) is not None:
                        log.info("time: %s was %+.0fs off; set to this machine's time", name, drift)
                elif cam.onvif_port and tz and name not in ignored:
                    result = await sync_onvif_zone(cam, tz)
                    if result:
                        log.info("time: %s time zone set to %s (%s)", name, zone, tz)
                    elif result is None:
                        ignored.add(name)
                        log.info("time: %s ignores time zones over ONVIF; its NTP keeps the clock right (in UTC)", name)
            except Exception as e:
                log.warning("time: %s: %s", name, e)
        await asyncio.sleep(tsc.interval_hours * 3600)
