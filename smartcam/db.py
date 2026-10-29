"""SQLite store for events and settings (away mode)."""

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
create table if not exists events (
    id          integer primary key autoincrement,
    camera      text not null,
    label       text not null,
    start_ts    real not null,
    end_ts      real,
    max_score   real not null,
    zones       text not null default '[]',
    score       real not null default 0,      -- suspicion score
    reasons     text not null default '[]',
    snapshot    text,
    clip        text,
    notified    integer not null default 0,
    review      text                           -- null, 'ok' or 'false_positive'
);
create index if not exists events_start on events (start_ts desc);
create table if not exists settings (key text primary key, value text not null);
"""

JSON_FIELDS = ("zones", "reasons", "box")


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        for column, kind in (("speed", "real"), ("crop", "text"), ("plate", "text"), ("plate_conf", "real"), ("box", "text"),
                             ("person", "text"), ("person_score", "real"), ("person_via", "text"), ("face", "text")):   # added later
            try:
                self.conn.execute(f"alter table events add column {column} {kind}")
            except sqlite3.OperationalError:
                pass  # already there
        self.lock = threading.Lock()

    def _row(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for f in JSON_FIELDS:
            d[f] = json.loads(d[f]) if d[f] is not None else None
        return d

    def insert_event(self, **fields) -> int:
        fields = {k: json.dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()}
        with self.lock, self.conn:
            cur = self.conn.execute(
                f"insert into events ({', '.join(fields)}) values ({', '.join('?' * len(fields))})",
                list(fields.values()))
            return cur.lastrowid

    def update_event(self, event_id: int, **fields) -> None:
        fields = {k: json.dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()}
        with self.lock, self.conn:
            self.conn.execute(f"update events set {', '.join(f'{k} = ?' for k in fields)} where id = ?",
                              [*fields.values(), event_id])

    def event(self, event_id: int) -> dict | None:
        with self.lock:
            return self._row(self.conn.execute("select * from events where id = ?", (event_id,)).fetchone())

    VEHICLES = ("car", "truck", "bus", "motorcycle", "bicycle")

    def events(self, limit: int = 100, camera: str | None = None, min_score: float = 0, kind: str | None = None,
               since: float | None = None, until: float | None = None) -> list[dict]:
        """Newest first. kind: person, vehicle or notified; since/until bound the start time."""
        sql, args = "select * from events where score >= ?", [min_score]
        if camera:
            sql += " and camera = ?"
            args.append(camera)
        if kind == "person":
            sql += " and label = 'person'"
        elif kind == "vehicle":
            sql += f" and label in ({','.join('?' * len(self.VEHICLES))})"
            args += self.VEHICLES
        elif kind == "doorbell":
            sql += " and label = 'doorbell'"
        elif kind == "smoke":
            sql += " and label = 'smoke'"
        elif kind == "notified":
            sql += " and notified = 1"
        if since is not None:
            sql += " and start_ts >= ?"
            args.append(since)
        if until is not None:
            sql += " and start_ts < ?"
            args.append(until)
        with self.lock:
            rows = self.conn.execute(sql + " order by start_ts desc limit ?", [*args, limit]).fetchall()
        return [self._row(r) for r in rows]

    def events_before(self, ts: float | None = None, limit: int = 1000) -> list[dict]:
        """Oldest events first; only those that started before `ts` if given."""
        sql, args = "select * from events", []
        if ts is not None:
            sql, args = sql + " where start_ts < ?", [ts]
        with self.lock:
            rows = self.conn.execute(sql + " order by start_ts limit ?", [*args, limit]).fetchall()
        return [self._row(r) for r in rows]

    def delete_event(self, event_id: int) -> None:
        with self.lock, self.conn:
            self.conn.execute("delete from events where id = ?", (event_id,))

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.lock:
            row = self.conn.execute("select value from settings where key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("insert into settings (key, value) values (?, ?) "
                              "on conflict(key) do update set value = excluded.value", (key, value))
