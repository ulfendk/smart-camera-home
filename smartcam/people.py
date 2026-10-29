"""The people SmartCam can name: everyone in Home Assistant (always family), plus people added by hand (friends,
neighbours, grandparents), and the faces it has learned for each.

Faces are stored per event as embeddings (faces.py), unnamed until someone names the event ("Who is this?").
Naming files them under that person, which is how SmartCam learns a face. Nothing leaves this machine, and
removing a person removes their faces.
"""

import threading

import numpy as np

from .db import Database

GROUPS = ("family", "friend", "neighbour", "other")

SCHEMA = """
create table if not exists people (
    name    text primary key,
    grp     text not null default 'other',
    source  text not null default 'manual',   -- 'ha' (a Home Assistant person) or 'manual'
    entity  text
);
create table if not exists faces (
    id        integer primary key autoincrement,
    event_id  integer not null,
    person    text,                           -- null until named
    embedding blob not null,                  -- float32, L2-normalised
    crop      text,                           -- face picture, relative to the data dir
    ts        real not null
);
create index if not exists faces_event on faces (event_id);
create index if not exists faces_person on faces (person);
"""


class People:
    def __init__(self, db: Database):
        self.db = db
        with db.lock:
            db.conn.executescript(SCHEMA)
        self._gallery: dict[str, np.ndarray] | None = None
        self.lock = threading.Lock()

    def _q(self, sql: str, args: tuple = ()) -> list:
        with self.db.lock:
            rows = self.db.conn.execute(sql, args).fetchall()
            self.db.conn.commit()
        return rows

    def all(self) -> list[dict]:
        rows = self._q("select p.name, p.grp, p.source, p.entity, count(f.id) from people p "
                       "left join faces f on f.person = p.name group by p.name order by p.grp != 'family', p.name")
        return [{"name": n, "group": g, "source": s, "entity": e, "faces": c} for n, g, s, e, c in rows]

    def group(self, name: str | None) -> str | None:
        rows = self._q("select grp from people where name = ?", (name,)) if name else []
        return rows[0][0] if rows else None

    def add(self, name: str, group: str = "other") -> None:
        if group not in GROUPS:
            raise ValueError(f"group must be one of {', '.join(GROUPS)}")
        name = name.strip()
        if not name:
            raise ValueError("a name is needed")
        if self._q("select 1 from people where name = ? and source = 'ha'", (name,)):
            raise ValueError(f"{name} comes from Home Assistant")
        self._q("insert into people (name, grp) values (?, ?) on conflict (name) do update set grp = excluded.grp",
                (name, group))

    def remove(self, name: str) -> None:
        if self._q("select 1 from people where name = ? and source = 'ha'", (name,)):
            raise ValueError(f"{name} comes from Home Assistant and stays (remove them there)")
        self._q("delete from people where name = ?", (name,))
        self.forget(name)

    def sync_ha(self, persons: dict[str, str]) -> None:
        """Home Assistant's people (entity -> name) are family, always; ones removed there go here too."""
        for entity, name in persons.items():
            self._q("insert into people (name, grp, source, entity) values (?, 'family', 'ha', ?) "
                    "on conflict (name) do update set grp = 'family', source = 'ha', entity = excluded.entity",
                    (name, entity))
        for (name,) in self._q("select name from people where source = 'ha'"):
            if name not in persons.values():
                self._q("delete from people where name = ?", (name,))

    # --- faces ---

    def add_face(self, event_id: int, embedding: np.ndarray, crop: str | None, ts: float, person: str | None = None) -> None:
        self._q("insert into faces (event_id, person, embedding, crop, ts) values (?, ?, ?, ?, ?)",
                (event_id, person, embedding.astype(np.float32).tobytes(), crop, ts))
        if person:
            self._gallery = None

    def event_faces(self, event_id: int) -> list[dict]:
        return [{"id": i, "person": p, "crop": c} for i, p, c in
                self._q("select id, person, crop from faces where event_id = ? order by id", (event_id,))]

    def name_event(self, event_id: int, name: str | None) -> None:
        """Teach SmartCam: the faces of this event are `name` (None: nobody, unlearn them)."""
        if name and not self._q("select 1 from people where name = ?", (name,)):
            raise ValueError(f"{name} is not on the people list")
        self._q("update faces set person = ? where event_id = ?", (name, event_id))
        self._gallery = None

    def forget(self, name: str) -> None:
        self._q("delete from faces where person = ?", (name,))
        self._gallery = None

    def gallery(self) -> dict[str, np.ndarray]:
        """name -> (n x 512) embeddings of everyone with learned faces."""
        with self.lock:
            if self._gallery is None:
                g: dict[str, list] = {}
                for name, blob in self._q("select f.person, f.embedding from faces f join people p on p.name = f.person"):
                    g.setdefault(name, []).append(np.frombuffer(blob, np.float32))
                self._gallery = {n: np.stack(v) for n, v in g.items()}
            return self._gallery
