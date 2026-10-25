"""Pick a balanced regression set from a copied Frigate database and list the recording segments it needs.

Usage: uv run python tools/select_regression_events.py [local/frigate]
Reads <dir>/frigate.db and writes:
  <dir>/needed.txt        segment paths relative to /frigate/media/recordings/ (for rsync --files-from)
  <dir>/regression.json   one case per event/window: camera, category, Frigate labels, time window, segments
"""

import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartcam.config import load_config  # noqa: E402

PRE, POST = 5.0, 5.0     # seconds of context around each event
MAX_WINDOW = 60.0        # long events (parked cars) only need their first minute
REC_PREFIX = "/media/frigate/recordings/"
VEHICLES = ("car", "bicycle", "motorcycle")

# category -> max cases per camera
CAPS = {
    "indoor_person": 120,        # lamp / sunlight suspects
    "person_near_vehicle": 25,   # person overlapping a car/bike event: the "wrong object wins" case
    "person": 12,
    "vehicle": 8,
    "quiet": 6,                  # motion but no objects: should stay silent
}


def load_events(db: sqlite3.Connection, since: float) -> list[dict]:
    rows = db.execute(
        "select id, camera, label, start_time, coalesce(end_time, start_time + 10), top_score, zones, data "
        "from event where start_time >= ? order by start_time", (since,))
    events = []
    for id_, camera, label, start, end, top_score, zones, data in rows:
        data = json.loads(data or "{}")
        events.append({
            "id": id_, "camera": camera, "label": label, "start": start, "end": end,
            "top_score": top_score or data.get("top_score"), "zones": json.loads(zones or "[]"),
            "box": data.get("box"), "path_data": data.get("path_data", []),
        })
    return events


def overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def categorise(events: list[dict], indoor: set[str]) -> dict[str, list[dict]]:
    vehicles_by_cam = defaultdict(list)
    for e in events:
        if e["label"] in VEHICLES:
            vehicles_by_cam[e["camera"]].append(e)

    cats = defaultdict(list)
    for e in events:
        if e["label"] == "person" and e["camera"] in indoor:
            cats["indoor_person"].append(e)
        elif e["label"] == "person":
            near = [v for v in vehicles_by_cam[e["camera"]] if overlaps(e, v)]
            if near:
                cats["person_near_vehicle"].append({**e, "overlapping": [v["id"] for v in near]})
            else:
                cats["person"].append(e)
        elif e["label"] in VEHICLES:
            cats["vehicle"].append(e)
    return cats


def quiet_windows(db: sqlite3.Connection, since: float) -> list[dict]:
    rows = db.execute(
        "select camera, start_time, end_time from recordings "
        "where start_time >= ? and motion > 0 and coalesce(objects, 0) = 0", (since,))
    return [{"id": f"quiet-{cam}-{start:.0f}", "camera": cam, "label": None,
             "start": start, "end": end + 10, "top_score": None, "zones": [], "box": None, "path_data": []}
            for cam, start, end in rows]


def segments_for(db: sqlite3.Connection, camera: str, start: float, end: float) -> list[tuple[str, float]]:
    rows = db.execute(
        "select path, segment_size from recordings where camera = ? and end_time > ? and start_time < ? "
        "order by start_time", (camera, start, end))
    return [(p.removeprefix(REC_PREFIX), size) for p, size in rows]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "local/frigate")
    db = sqlite3.connect(root / "frigate.db")
    since = db.execute("select min(start_time) from recordings").fetchone()[0]
    indoor = {name for name, cam in load_config().cameras.items() if cam.location == "indoor"}

    cats = categorise(load_events(db, since), indoor)
    cats["quiet"] = quiet_windows(db, since)

    rng = random.Random(42)
    cases, needed, total_mb = [], {}, 0.0
    for cat, cap in CAPS.items():
        by_cam = defaultdict(list)
        for e in cats.get(cat, []):
            by_cam[e["camera"]].append(e)
        for camera, evs in sorted(by_cam.items()):
            for e in rng.sample(evs, min(cap, len(evs))):
                start = e["start"] - PRE
                end = min(e["end"], e["start"] + MAX_WINDOW) + POST
                segs = segments_for(db, camera, start, end)
                if not segs:
                    continue
                for path, size in segs:
                    if path not in needed:
                        needed[path] = size
                        total_mb += size
                cases.append({**e, "category": cat, "window": [start, end], "segments": [p for p, _ in segs]})

    cases.sort(key=lambda c: (c["camera"], c["window"][0]))
    (root / "needed.txt").write_text("\n".join(sorted(needed)) + "\n")
    (root / "regression.json").write_text(json.dumps(cases, indent=1))

    summary = defaultdict(lambda: defaultdict(int))
    for c in cases:
        summary[c["category"]][c["camera"]] += 1
    for cat, cams in summary.items():
        print(f"{cat:20} " + ", ".join(f"{cam}={n}" for cam, n in sorted(cams.items())))
    print(f"\n{len(cases)} cases, {len(needed)} segments, ~{total_mb / 1024:.1f} GB")
    print(f"Wrote {root / 'needed.txt'} and {root / 'regression.json'}")


if __name__ == "__main__":
    main()
