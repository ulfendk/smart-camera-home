"""Dahua RPC2 (JSON over HTTP), for what the CGI API refuses on these cameras: the deterrence outputs.

The red/blue warning light ("WhiteLight" in CoaxialControlIO) is switched with CoaxialControlIO.control. The camera's own web UI sends exactly these calls. Login is Dahua's
two-step challenge (from the web UI's rpcCore.js; the Home Assistant Dahua integration does the same).
"""

import hashlib
import logging

import httpx

log = logging.getLogger(__name__)

WARNING_LIGHT = 1   # CoaxialControlIO type of the red/blue warning light


class RpcError(Exception):
    pass


class DahuaRpc:
    def __init__(self, ip: str, user: str, password: str, timeout: float = 8):
        self.base, self.user, self.password = f"http://{ip}", user, password
        self.http = httpx.AsyncClient(timeout=timeout)
        self.session = None
        self.id = 0

    async def _post(self, path: str, method: str, params=None) -> dict:
        self.id += 1
        data = {"method": method, "id": self.id}
        if params is not None:
            data["params"] = params
        if self.session:
            data["session"] = self.session
        r = await self.http.post(self.base + path, json=data)
        r.raise_for_status()
        return r.json()

    async def login(self) -> None:
        self.session = None
        r = await self._post("/RPC2_Login", "global.login", {"userName": self.user, "password": "", "clientType": "Web5.0"})
        self.session, p = r["session"], r["params"]
        h1 = hashlib.md5(f"{self.user}:{p['realm']}:{self.password}".encode()).hexdigest().upper()
        h2 = hashlib.md5(f"{self.user}:{p['random']}:{h1}".encode()).hexdigest().upper()
        r = await self._post("/RPC2_Login", "global.login", {
            "userName": self.user, "password": h2, "clientType": "Web5.0", "realm": p["realm"], "random": p["random"],
            "passwordType": "Default", "authorityType": p.get("encryption") or "Default"})
        if not r.get("result"):
            raise RpcError(f"login refused: {r.get('error')}")
        self.session = r["session"]

    async def call(self, method: str, params=None) -> dict:
        """One call; logs in first, and once more if the session has expired."""
        for attempt in range(2):
            if self.session is None:
                await self.login()
            r = await self._post("/RPC2", method, params)
            if r.get("result") is not False:
                return r.get("params") or {}
            err = r.get("error") or {}
            if attempt == 0 and err.get("code") in (287637505, 287637504):   # invalid / expired session
                self.session = None
                continue
            raise RpcError(f"{method} refused: code {err.get('code')} {err.get('message', '')}".strip())

    async def output(self, kind: int, on: bool) -> None:
        await self.call("CoaxialControlIO.control",
                        {"channel": 0, "info": [{"Type": kind, "IO": 1 if on else 2, "TriggerMode": 2}]})

    async def status(self) -> dict:
        return (await self.call("CoaxialControlIO.getStatus", {"channel": 0})).get("status", {})

    async def close(self) -> None:
        await self.http.aclose()
