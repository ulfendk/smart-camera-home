"""Keep an external process (go2rtc, ffmpeg) running, restarting it with backoff when it exits."""

import asyncio
import logging
import re
import time
from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)


CREDENTIALS = re.compile(r"(\w+://)[^/@\s:]+:[^/@\s]+@")


def redact(text: str) -> str:
    """ffmpeg echoes its input URL in errors; keep camera and restream passwords out of the log."""
    return CREDENTIALS.sub(r"\1***@", text)


async def supervise(name: str, argv: Callable[[], Sequence[str]], *, max_backoff: float = 60) -> None:
    backoff = 1.0
    while True:
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        log.info("%s started (pid %d)", name, proc.pid)
        try:
            _, stderr = await proc.communicate()
        except asyncio.CancelledError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except TimeoutError:
                proc.kill()
            raise
        tail = stderr.decode(errors="replace").strip().splitlines()[-3:]
        # A process that ran for a while gets a fresh backoff; one that dies at once backs off further.
        backoff = 1.0 if time.monotonic() - started > 60 else min(backoff * 2, max_backoff)
        log.warning("%s exited with %s; restarting in %.0fs. %s", name, proc.returncode, backoff, redact(" | ".join(tail)))
        await asyncio.sleep(backoff)
