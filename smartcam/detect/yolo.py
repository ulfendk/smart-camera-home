"""YOLO (Ultralytics ONNX export, 84 x N output) on ONNX Runtime: CUDA on Linux, CoreML on macOS, else CPU."""

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from . import Detection

log = logging.getLogger(__name__)

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()

# Classes we care about; everything else is dropped right after inference.
TRACKED = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "cat", "dog"}
PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")


class YoloDetector:
    def __init__(self, model: Path, min_score: float = 0.3, iou: float = 0.5):
        providers = [p for p in PREFERRED_PROVIDERS if p in ort.get_available_providers()]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # CoreML logs partitioning warnings on every load
        self.session = ort.InferenceSession(str(model), sess_options=opts, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[2]
        self.min_score, self.iou = min_score, iou
        self.wanted = np.array([c in TRACKED for c in COCO])
        # Camera threads share one session. Concurrent runs crash the CUDA provider (cuDNN failures,
        # illegal memory access), and one GPU gains nothing from them; pre/post-processing stays parallel.
        self.lock = threading.Lock()
        log.info("%s on %s", model.name, self.session.get_providers()[0])

    def letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, scale, left, top

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect tracked classes in a BGR image. Boxes come back normalised to the image."""
        h, w = image.shape[:2]
        canvas, scale, left, top = self.letterbox(image)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255
        with self.lock:
            out = self.session.run(None, {self.input.name: blob})[0][0].T  # N x 84: cx, cy, w, h, 80 class scores

        scores = out[:, 4:] * self.wanted
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= self.min_score
        if not keep.any():
            return []
        boxes, cls, best = out[keep, :4], cls[keep], best[keep]
        xywh = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]])
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), best.tolist(), cls.tolist(), self.min_score, self.iou)

        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            x1, y1 = (x - left) / scale / w, (y - top) / scale / h
            x2, y2 = (x + bw - left) / scale / w, (y + bh - top) / scale / h
            box = tuple(float(np.clip(v, 0, 1)) for v in (x1, y1, x2, y2))
            dets.append(Detection(COCO[cls[i]], float(best[i]), box))
        return dets
