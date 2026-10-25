"""Replay the labelled Frigate cases through the smartcam pipeline and score person alerts.

Usage: uv run python tools/run_regression.py [--model models/yolo11m.onnx] [--category C] [--limit N]

Needs local/frigate/{frigate.db,regression.json,recordings/} (see select_regression_events.py) and
local/frigate/labels.json (from contact_sheet.py). Writes local/frigate/regression_out/results.json, plus an annotated
frame for every case where the outcome disagrees with the label.
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartcam.config import ROOT, load_config  # noqa: E402
from smartcam.detect.yolo import YoloDetector  # noqa: E402
from smartcam.ingest import file_frames  # noqa: E402
from smartcam.pipeline import CameraPipeline  # noqa: E402

FRIGATE = ROOT / "local" / "frigate"
OUT = FRIGATE / "regression_out"


def annotate(image, tracks, title: str, ts: float | None = None):
    img = image.copy()
    h, w = img.shape[:2]
    for t in tracks:
        if ts is not None and t.last_ts != ts:
            continue  # not seen in this frame; its box would be stale
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(t.box, (w, h, w, h)))
        color = (0, 0, 255) if t.label == "person" else (255, 160, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f"{t.label} {t.max_score:.2f} tr={t.travel():.2f}{' S' if t.stationary else ''}{' M' if t.moving else ''}"
        cv2.putText(img, tag, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolo11m.onnx")
    ap.add_argument("--category")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = load_config()
    cases = json.loads((FRIGATE / "regression.json").read_text())
    labels = {l["idx"]: l for l in json.loads((FRIGATE / "labels.json").read_text())}
    db = sqlite3.connect(FRIGATE / "frigate.db")
    detector = YoloDetector(ROOT / args.model)
    OUT.mkdir(parents=True, exist_ok=True)

    selected = [(i, c) for i, c in enumerate(cases) if not args.category or c["category"] == args.category]
    if args.limit:
        selected = selected[:args.limit]

    results, t0 = [], time.time()
    for n, (idx, case) in enumerate(selected, 1):
        files = []
        for seg in case["segments"]:
            row = db.execute("select start_time from recordings where path = ?",
                             ("/media/frigate/recordings/" + seg,)).fetchone()
            files.append((row[0], FRIGATE / "recordings" / seg))
        pipeline = CameraPipeline(cfg.cameras[case["camera"]], detector)
        start, end = case["window"]
        confirmed, snap, last_view, frames = [], None, None, 0
        for frame in file_frames(case["camera"], files, cfg.detect.fps, cfg.detect.max_width, cfg.detect.hwaccel):
            res = pipeline.process(frame)
            frames += 1
            # Confirmations in the pre-roll before the window count too: the person was already
            # there when Frigate's event started.
            if frame.ts <= end:
                for t in res.confirmed:
                    confirmed.append({"label": t.label, "score": round(t.max_score, 2), "travel": round(t.travel(), 2),
                                      "ts": round(frame.ts - start, 1)})
                    if snap is None:
                        snap = annotate(frame.image, res.tracks, f"#{idx} confirmed {t.label}", frame.ts)
            if snap is None and res.detections:
                last_view = (frame.image, res.tracks, f"#{idx}", frame.ts)

        truth = labels[idx]["truth"]
        alerted = any(c["label"] == "person" for c in confirmed)
        frigate_alerted = case["label"] == "person"
        outcome = ("hit" if alerted else "miss") if truth == "person" else ("false_alarm" if alerted else "correct_reject")
        if truth == "unsure":
            outcome = "unsure"
        results.append({"idx": idx, "camera": case["camera"], "category": case["category"], "truth": truth,
                        "frigate_label": case["label"], "frigate_alerted": frigate_alerted,
                        "outcome": outcome, "confirmed": confirmed, "frames": frames})
        if outcome in ("miss", "false_alarm"):
            img = snap if snap is not None else last_view and annotate(last_view[0], last_view[1], f"#{idx} {outcome}", last_view[3])
            if img is not None:
                cv2.imwrite(str(OUT / f"{idx:03d}_{outcome}.jpg"), img)
        rate = (time.time() - t0) / n
        print(f"[{n}/{len(selected)}] #{idx} {case['camera']:10} {case['category']:20} truth={truth:7} -> {outcome}"
              f"  ({rate:.1f}s/case)", flush=True)

    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    summarise(results)


def summarise(results: list[dict]) -> None:
    print("\n== outcome by category")
    by_cat = Counter((r["category"], r["outcome"]) for r in results)
    for cat in sorted({r["category"] for r in results}):
        print(f"  {cat:20} " + ", ".join(f"{o}={n}" for (c, o), n in sorted(by_cat.items()) if c == cat))

    persons = [r for r in results if r["truth"] == "person"]
    others = [r for r in results if r["truth"] not in ("person", "unsure")]
    ours_fa = sum(r["outcome"] == "false_alarm" for r in others)
    frigate_fa = sum(r["frigate_alerted"] for r in others)
    hits = sum(r["outcome"] == "hit" for r in persons)
    print(f"\n== person alerts   smartcam vs Frigate")
    print(f"  recall on real people:      {hits}/{len(persons)}   vs {sum(r['frigate_alerted'] for r in persons)}/{len(persons)}"
          "  (biased towards Frigate: most cases were sampled from its events)")
    print(f"  false alarms (no person):   {ours_fa}/{len(others)}   vs {frigate_fa}/{len(others)}")


if __name__ == "__main__":
    main()
