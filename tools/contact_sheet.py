"""Tile regression-case thumbnails into numbered contact sheets for quick labelling.

Usage: uv run python tools/contact_sheet.py [local/frigate]
Writes <dir>/sheets/sheet_NN.jpg. Each tile shows the case index (its position in regression.json),
camera, Frigate label and top score.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

COLS, ROWS, TILE = 6, 5, 175
CAPTION = 34


def tile(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = np.zeros((TILE + CAPTION, TILE, 3), np.uint8)
    out[CAPTION:] = cv2.resize(img, (TILE, TILE))
    for i, text in enumerate(lines):
        cv2.putText(out, text, (3, 14 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "local/frigate")
    cases = json.loads((root / "regression.json").read_text())
    tiles = []
    for idx, c in enumerate(cases):
        thumb = root / "thumbs" / c["camera"] / f"{c['id']}.webp"
        img = cv2.imread(str(thumb)) if c["label"] else None
        if img is None:
            continue
        score = f"{c['top_score']:.2f}" if c["top_score"] else "?"
        tiles.append(tile(img, [f"#{idx} {c['camera']}", f"{c['label']} {score} {c['category'][:11]}"]))

    out_dir = root / "sheets"
    out_dir.mkdir(exist_ok=True)
    per_sheet = COLS * ROWS
    blank = np.zeros_like(tiles[0])
    for s in range(0, len(tiles), per_sheet):
        chunk = tiles[s:s + per_sheet] + [blank] * (per_sheet - len(tiles[s:s + per_sheet]))
        rows = [np.hstack(chunk[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        cv2.imwrite(str(out_dir / f"sheet_{s // per_sheet:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(tiles)} tiles -> {out_dir} ({-(-len(tiles) // per_sheet)} sheets)")


if __name__ == "__main__":
    main()
