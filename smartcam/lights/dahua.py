"""Dahua white light ("Full-Color" / active deterrence warm light) over the camera's HTTP API.

On: every Lighting_V2 profile (day / night / normal) is switched to Manual at full brightness.
Off: the original modes and brightness are restored. The originals are saved to disk before the
first change, so a crash with the light on is repaired on the next start.
"""

import json
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

PROFILES = (0, 1, 2)


def parse_table(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.removeprefix("table.").strip()] = value.strip()
    return out


class DahuaLight:
    def __init__(self, name: str, ip: str, user: str, password: str, state_dir: Path, dry_run: bool = True):
        self.name, self.ip, self.dry_run = name, ip, dry_run
        self.auth = httpx.DigestAuth(user, password)
        self.state_file = state_dir / f"light_{name}.json"
        self.is_on = False

    async def _get(self, query: str) -> str:
        async with httpx.AsyncClient(auth=self.auth, timeout=5) as client:
            r = await client.get(f"http://{self.ip}/cgi-bin/{query}")
            r.raise_for_status()
            return r.text

    async def _set(self, values: dict[str, str]) -> None:
        query = "&".join(f"{k}={v}" for k, v in values.items())
        if self.dry_run:
            log.info("light %s (dry run): setConfig %s", self.name, query)
            return
        body = await self._get(f"configManager.cgi?action=setConfig&{query}")
        if "OK" not in body:
            raise RuntimeError(f"light {self.name}: setConfig failed: {body.strip()[:200]}")

    async def _originals(self) -> dict[str, str]:
        table = parse_table(await self._get("configManager.cgi?action=getConfig&name=Lighting_V2"))
        keys = [f"Lighting_V2[0][{p}][0].{f}" for p in PROFILES for f in ("Mode", "MiddleLight[0].Light")]
        return {k: table[k] for k in keys if k in table}

    async def restore_after_crash(self) -> None:
        if self.state_file.exists():
            log.warning("light %s: restoring settings left over from an unclean stop", self.name)
            await self._set(json.loads(self.state_file.read_text()))
            self.state_file.unlink()

    async def on(self) -> None:
        if self.is_on:
            return
        if not self.state_file.exists():
            originals = await self._originals()
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(originals))
        await self._set({k: v for p in PROFILES for k, v in (
            (f"Lighting_V2[0][{p}][0].Mode", "Manual"), (f"Lighting_V2[0][{p}][0].MiddleLight[0].Light", "100"))})
        self.is_on = True
        log.info("light %s on", self.name)

    async def off(self) -> None:
        if not self.is_on:
            return
        if self.state_file.exists():
            await self._set(json.loads(self.state_file.read_text()))
            self.state_file.unlink()
        self.is_on = False
        log.info("light %s off", self.name)
