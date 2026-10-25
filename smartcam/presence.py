"""Presence simulation while the alarm is armed, replaying the home's own evenings.

Learning: every few hours the light history Home Assistant still holds (the recorder keeps about ten
days) is turned into "evenings" and kept in SmartCam's own database, so the library grows for as long
as SmartCam runs. Only days when people were home count: with `ha.alarm_entity` set, a day whose
evening was armed is skipped, so the simulation never learns from an empty house or from itself.
Motion-sensor flicker is merged into visits and very short blips are dropped.

Replaying: each armed day, one real evening of the same kind is picked (Monday-Thursday, Friday,
Saturday, Sunday; a public holiday replays the latest Sunday) and all its lights are replayed together, shifted by up to `jitter_minutes` as a whole and `light_jitter_minutes`
per light, so rooms keep their real order. Times are warped to the season: sunrise and sunset of the
recorded day map to today's, while midnight, midday and `bedtime` stay put, and everything between
moves smoothly; lights that came on 20 minutes after dusk three weeks ago come on 20 minutes after
today's dusk. A light someone switches by hand is left alone for the rest of the day; on disarm only
the lights the simulation switched on are switched off.
"""

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import holidays as holiday_calendars
import httpx
from pydantic import BaseModel

from .config import Config
from .db import Database

log = logging.getLogger(__name__)

DAY = 86400


class PresenceConfig(BaseModel):
    enabled: bool = False
    lights: list[str] = []
    jitter_minutes: float = 20          # the whole evening moves by up to this
    light_jitter_minutes: float = 3     # each switch moves by up to this on top
    merge_gap_minutes: float = 3        # off-gaps shorter than this are one visit (motion lights)
    min_on_minutes: float = 1           # shorter blips are dropped
    max_on_hours: float = 4             # longer intervals are cut short
    bedtime: str = "23:00"              # stays at clock time when the season changes
    daylight_margin_minutes: float = 30 # intervals entirely in daylight (this far from sunrise/sunset) are skipped
    holidays_country: str | None = None # public holidays count as Sundays; default: Home Assistant's country
    extra_holidays: list[str] = []      # more days treated like holidays, as MM-DD (e.g. 12-24, 12-31)


@dataclass
class Action:
    ts: float
    light: str
    on: bool


# --- sun --------------------------------------------------------------------------------------------

def sun_times(day: date, lat: float, lon: float, tz: ZoneInfo) -> tuple[float, float]:
    """Local sunrise and sunset as seconds after midnight (NOAA approximation, about a minute accurate).
    Polar day/night fall back to 06:00 / 18:00."""
    n = day.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma) - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(math.radians(lat)) * math.cos(decl))
              - math.tan(math.radians(lat)) * math.tan(decl))
    if abs(cos_ha) > 1:
        return 6 * 3600.0, 18 * 3600.0
    ha = math.degrees(math.acos(cos_ha))
    noon = datetime(day.year, day.month, day.day, 12, tzinfo=tz)
    offset_min = noon.utcoffset().total_seconds() / 60
    rise = (720 - 4 * (lon + ha) - eqtime + offset_min) * 60
    sset = (720 - 4 * (lon - ha) - eqtime + offset_min) * 60
    return rise, sset


def at(day: date, clock: float, tz: ZoneInfo) -> float:
    """Timestamp of a local clock time (seconds after 00:00) on `day`. Clock time, not elapsed time,
    so the days the clocks change (23 or 25 hours long) come out right."""
    return (datetime(day.year, day.month, day.day, tzinfo=tz) + timedelta(seconds=clock)).timestamp()


def clock_of(ts: float, day: date, tz: ZoneInfo) -> float:
    """The local clock time of `ts`, in seconds after 00:00 of `day` (86400 and up for after midnight)."""
    dt = datetime.fromtimestamp(ts, tz)
    return (dt.date() - day).days * DAY + dt.hour * 3600 + dt.minute * 60 + dt.second


def _clock(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h) * 3600 + int(m) * 60


def warp(t: float, src: tuple[float, float], dst: tuple[float, float], bedtime: float) -> float:
    """Map a time of day on the recorded day to today: sunrise→sunrise, sunset→sunset, fixed
    midnight / midday / bedtime, piecewise linear in between."""
    pairs = [(0.0, 0.0), (src[0], dst[0]), (43200.0, 43200.0), (src[1], dst[1]), (bedtime, bedtime), (DAY, DAY)]
    pairs.sort()
    clean = [pairs[0]]
    for a, b in pairs[1:]:
        if a > clean[-1][0] and b > clean[-1][1]:   # keep it monotonic (e.g. a summer sunset after bedtime)
            clean.append((a, b))
    for (a0, b0), (a1, b1) in zip(clean, clean[1:]):
        if a0 <= t <= a1:
            return b0 + (t - a0) * (b1 - b0) / (a1 - a0)
    return t


# --- learning ---------------------------------------------------------------------------------------

def intervals(changes: list[tuple[float, bool]], start: float, end: float, pc: PresenceConfig) -> list[tuple[float, float]]:
    """On-intervals within [start, end) from a light's state changes, cleaned up, as seconds after `start`."""
    out, on_since, state = [], None, False
    for ts, on in sorted(changes):
        if ts >= end:
            break
        if on and not state:
            on_since = max(ts, start)
        elif not on and state and on_since is not None:
            if ts > start:
                out.append((on_since, ts))
            on_since = None
        state = on
    if state and on_since is not None:
        out.append((on_since, end))
    merged = []
    for s, e in out:
        if merged and s - merged[-1][1] < pc.merge_gap_minutes * 60:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s - start, min(e, s + pc.max_on_hours * 3600) - start)
            for s, e in merged if e - s >= pc.min_on_minutes * 60]


class Library:
    """Learned home evenings, kept in SmartCam's database (outlives Home Assistant's recorder)."""

    def __init__(self, db: Database):
        self.db = db
        with db.lock, db.conn:
            db.conn.execute("create table if not exists presence_days (day text primary key, weekend integer, "
                            "sunrise real, sunset real, lights text)")

    def days(self) -> list[dict]:
        with self.db.lock:
            rows = self.db.conn.execute("select day, weekend, sunrise, sunset, lights from presence_days order by day").fetchall()
        return [{"day": r[0], "weekend": bool(r[1]), "sun": (r[2], r[3]), "lights": json.loads(r[4])} for r in rows]

    def has(self, day: str) -> bool:
        with self.db.lock:
            return self.db.conn.execute("select 1 from presence_days where day = ?", (day,)).fetchone() is not None

    def add(self, day: date, sun: tuple[float, float], lights: dict) -> None:
        with self.db.lock, self.db.conn:
            self.db.conn.execute("insert or replace into presence_days values (?, ?, ?, ?, ?)",
                                 (day.isoformat(), int(day.weekday() >= 5), sun[0], sun[1], json.dumps(lights)))


# --- planning ---------------------------------------------------------------------------------------

FALLBACK = {"sunday": ["saturday"], "saturday": ["sunday"], "friday": ["weekday"], "weekday": ["friday"]}


def holiday_set(country: str | None, extra: list[str], years: range) -> set[date]:
    days = set(holiday_calendars.country_holidays(country, years=list(years)).keys()) if country else set()
    for md in extra:
        m, d = (int(x) for x in md.split("-"))
        days |= {date(y, m, d) for y in years}
    return days


def kind(day: date, hols: set[date]) -> str:
    """Which evenings are alike: holidays live like Sundays."""
    if day in hols or day.weekday() == 6:
        return "sunday"
    return {5: "saturday", 4: "friday"}.get(day.weekday(), "weekday")


def plan(day: date, library: list[dict], pc: PresenceConfig, sun: tuple[float, float], tz: ZoneInfo,
         rng: random.Random, hols: set[date] = frozenset()) -> tuple[str | None, list[Action]]:
    """The switching plan for `day`: (source day, actions). Empty without a learned evening."""
    if not library:
        return None, []
    want = kind(day, hols)
    of = lambda d: kind(date.fromisoformat(d["day"]), hols)
    pool = [d for d in library if of(d) == want]
    for other in FALLBACK[want]:
        pool = pool or [d for d in library if of(d) == other]
    pool = pool or library
    # A holiday replays the latest Sunday; other days a random evening of their kind.
    src = max(pool, key=lambda d: d["day"]) if day in hols else rng.choice(pool)
    shift = rng.uniform(-pc.jitter_minutes, pc.jitter_minutes) * 60
    bedtime = _clock(pc.bedtime)
    margin = pc.daylight_margin_minutes * 60
    actions = []
    for light, spans in src["lights"].items():
        if pc.lights and light not in pc.lights:
            continue
        for s, e in spans:
            a = warp(s, src["sun"], sun, bedtime) + shift + rng.uniform(-1, 1) * pc.light_jitter_minutes * 60
            b = warp(e, src["sun"], sun, bedtime) + shift + rng.uniform(-1, 1) * pc.light_jitter_minutes * 60
            if b <= a:
                continue
            if a >= sun[0] + margin and b <= sun[1] - margin:
                continue   # entirely in daylight: nobody notices, nothing to simulate
            actions += [Action(at(day, a, tz), light, True), Action(at(day, b, tz), light, False)]
    return src["day"], sorted(actions, key=lambda x: x.ts)


# --- running ----------------------------------------------------------------------------------------

class PresenceSimulator:
    def __init__(self, cfg: Config, db: Database, alarm, pc: PresenceConfig | None = None):
        self.cfg, self.alarm = cfg, alarm
        self.pc = pc or PresenceConfig(**cfg.presence)
        self.library = Library(db)
        self.http = httpx.AsyncClient(base_url=cfg.ha.url, timeout=30, headers={"Authorization": f"Bearer {cfg.ha.token}"})
        self.where: tuple[float, float, ZoneInfo] | None = None
        self.ours: dict[str, bool] = {}       # light -> the state we last set
        self.manual: set[str] = set()         # lights someone took over; left alone today
        self.current: tuple[str | None, list[Action]] = (None, [])
        self.country: str | None = self.pc.holidays_country

    async def _where(self) -> tuple[float, float, ZoneInfo]:
        if self.where is None:
            c = (await self.http.get("/api/config")).json()
            self.where = (c["latitude"], c["longitude"], ZoneInfo(c["time_zone"]))
            self.country = self.country or c.get("country")
        return self.where

    def holidays(self, around: date) -> set[date]:
        return holiday_set(self.country, self.pc.extra_holidays, range(around.year - 1, around.year + 2))

    async def sun(self, day: date) -> tuple[float, float]:
        lat, lon, tz = await self._where()
        return sun_times(day, lat, lon, tz)

    async def learn(self) -> int:
        """Add the complete past days Home Assistant still has to the library. Returns how many."""
        lat, lon, tz = await self._where()
        entities = list(self.pc.lights) + ([self.cfg.ha.alarm_entity] if self.cfg.ha.alarm_entity else [])
        if not self.pc.lights:
            return 0
        now = datetime.now(tz)
        start = (now - timedelta(days=12)).replace(hour=0, minute=0, second=0, microsecond=0)
        r = await self.http.get(f"/api/history/period/{start.isoformat()}", params={
            "filter_entity_id": ",".join(entities), "end_time": now.isoformat(), "minimal_response": "1", "no_attributes": "1"})
        r.raise_for_status()
        changes: dict[str, list[tuple[float, bool]]] = {}
        for series in r.json():
            eid = series[0]["entity_id"]
            changes[eid] = [(datetime.fromisoformat(s["last_changed"]).timestamp(), s["state"] == "on") for s in series
                            if s["state"] in ("on", "off")]
        added = 0
        day = start.date() + timedelta(days=1)   # the first day may start before the history does
        while day < now.date():
            if not self.library.has(day.isoformat()):
                start_ts, end_ts = at(day, 0, tz), at(day + timedelta(days=1), 0, tz)
                if self._home(changes.get(self.cfg.ha.alarm_entity), day, tz):
                    lights = {l: [(clock_of(start_ts + s, day, tz), clock_of(start_ts + e, day, tz))
                                  for s, e in intervals(changes.get(l, []), start_ts, end_ts, self.pc)]
                              for l in self.pc.lights}
                    if any(lights.values()):
                        self.library.add(day, sun_times(day, lat, lon, tz), lights)
                        added += 1
            day += timedelta(days=1)
        return added

    def _home(self, alarm: list[tuple[float, bool]] | None, day: date, tz: ZoneInfo) -> bool:
        """Was the house lived in that evening? Not if the alarm was on for over half an hour of 15:00-24:00."""
        if alarm is None:
            return True
        armed = sum(e - s for s, e in intervals(alarm, at(day, 15 * 3600, tz), at(day + timedelta(days=1), 0, tz),
                                               PresenceConfig(merge_gap_minutes=0, min_on_minutes=0, max_on_hours=24)))
        return armed < 1800

    async def preview(self, day: date | None = None) -> dict:
        lat, lon, tz = await self._where()
        day = day or datetime.now(tz).date()
        sun = sun_times(day, lat, lon, tz)
        hols = self.holidays(day)
        src, actions = plan(day, self.library.days(), self.pc, sun, tz, random.Random(day.toordinal()), hols)
        fmt = lambda ts: datetime.fromtimestamp(ts, tz).strftime("%H:%M")
        return {"day": day.isoformat(), "kind": kind(day, hols), "from": src,
                "sunrise": fmt(at(day, sun[0], tz)), "sunset": fmt(at(day, sun[1], tz)),
                "learned_days": len(self.library.days()),
                "actions": [{"at": fmt(a.ts), "light": a.light, "on": a.on} for a in actions]}

    async def _state(self, light: str) -> bool | None:
        r = await self.http.get(f"/api/states/{light}")
        return r.json().get("state") == "on" if r.status_code == 200 else None

    async def _switch(self, light: str, on: bool) -> None:
        if light in self.manual:
            return
        now_on = await self._state(light)
        if light in self.ours and now_on is not None and now_on != self.ours[light]:
            log.info("presence: %s was switched by hand; leaving it alone today", light)
            self.manual.add(light)
            return
        if now_on == on:
            self.ours.setdefault(light, on)   # already as planned; not ours to switch off later
            return
        await self.http.post(f"/api/services/light/{'turn_on' if on else 'turn_off'}", json={"entity_id": light})
        self.ours[light] = on

    async def _stop(self) -> None:
        for light, on in list(self.ours.items()):
            if on and light not in self.manual and await self._state(light):
                await self.http.post("/api/services/light/turn_off", json={"entity_id": light})
        self.ours.clear()
        self.manual.clear()

    @property
    def active(self) -> bool:
        """Switching lights: only when enabled (with lights alone it learns and previews)."""
        return self.pc.enabled and self.alarm is not None and self.alarm.state in ("armed_away", "pending", "triggered")

    async def run(self) -> None:
        last_learn = 0.0
        while True:
            try:
                if time.time() - last_learn > 6 * 3600:
                    last_learn = time.time()
                    if n := await self.learn():
                        log.info("presence: learned %d more evening(s); %d in the library", n, len(self.library.days()))
                if self.active:
                    await self._simulate()
            except (httpx.HTTPError, KeyError, ValueError) as e:
                log.warning("presence simulation: %s", e)
            await asyncio.sleep(30)

    async def _simulate(self) -> None:
        """Run today's plan while armed; stop (and tidy up) when disarmed or at midnight."""
        lat, lon, tz = await self._where()
        today = datetime.now(tz).date()
        sun = sun_times(today, lat, lon, tz)
        src, actions = plan(today, self.library.days(), self.pc, sun, tz, random.Random(today.toordinal()),
                            self.holidays(today))
        if src is None:
            log.info("presence: nothing learned yet, so nothing to replay")
            while self.active and datetime.now(tz).date() == today:
                await asyncio.sleep(60)
            return
        log.info("presence: replaying the evening of %s (%d switches today)", src, len(actions))
        now = time.time()
        # Lights that should already be on at this moment are switched on straight away.
        for light in {a.light for a in actions}:
            past = [a for a in actions if a.light == light and a.ts <= now]
            if past and past[-1].on:
                await self._switch(light, True)
        try:
            for a in [a for a in actions if a.ts > now]:
                while self.active and time.time() < a.ts:
                    await asyncio.sleep(min(30, a.ts - time.time()))
                if not self.active:
                    break
                await self._switch(a.light, a.on)
            while self.active and datetime.now(tz).date() == today:
                await asyncio.sleep(60)
        finally:
            if not self.active:
                await self._stop()
