"""Check that the Coral USB works: load the Edge TPU delegate, time inference, print detections.

Usage: uv run python tools/check_coral.py [image.jpg]
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "bin" / "lib" / "libedgetpu.1.dylib"
MODEL = ROOT / "models" / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 15: "bird", 16: "cat", 17: "dog"}


def main() -> None:
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "data" / "probe").glob("*.jpg"))[0]
    t0 = time.perf_counter()
    interp = Interpreter(str(MODEL), experimental_delegates=[load_delegate(str(LIB))])
    interp.allocate_tensors()
    print(f"delegate + model loaded in {time.perf_counter() - t0:.1f}s")

    inp = interp.get_input_details()[0]
    _, h, w, _ = inp["shape"]
    frame = cv2.imread(str(image))
    rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
    interp.set_tensor(inp["index"], rgb[np.newaxis].astype(np.uint8))

    times = []
    for _ in range(50):
        t = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t) * 1000)
    print(f"inference {w}x{h}: median {np.median(times):.1f} ms, first {times[0]:.1f} ms")

    out = interp.get_output_details()
    boxes, classes, scores = (interp.get_tensor(out[i]["index"])[0] for i in range(3))
    for box, cls, score in zip(boxes, classes, scores):
        if score >= 0.4:
            print(f"  {COCO.get(int(cls), int(cls))}: {score:.2f} box={np.round(box, 2).tolist()}")


if __name__ == "__main__":
    main()
