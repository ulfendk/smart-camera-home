# smartcam for Linux: CUDA inference on an NVIDIA GPU, VAAPI decoding on an Intel iGPU.
# Built by .github/workflows/image.yml and published to GitHub Container Registry.
# The config is not in the image: it is read from /app/data/smartcam.yml (the mounted data dir).

# --- YOLO11 exported to ONNX (needs torch, so it stays out of the runtime image) ---
FROM python:3.12-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "ultralytics==8.3.*" onnx onnxslim onnxruntime
WORKDIR /models
RUN for m in yolo11s yolo11m; do \
        yolo export model=$m.pt format=onnx imgsz=640 simplify=True || exit 1; \
    done && rm -f *.pt

# --- runtime ---
# CUDA 12, not 13, so Pascal GPUs (GTX 10-series) still work. Needs an NVIDIA driver >= 525 on the host.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG GO2RTC_VERSION=v1.9.14
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python LIBVA_DRIVER_NAME=iHD

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg intel-media-va-driver-non-free vainfo ca-certificates curl tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/bin \
    && curl -fsSL -o /app/bin/go2rtc \
        "https://github.com/AlexxIT/go2rtc/releases/download/${GO2RTC_VERSION}/go2rtc_linux_amd64" \
    && chmod +x /app/bin/go2rtc

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project -p 3.12
COPY --from=models /models/*.onnx models/
COPY smartcam smartcam
COPY config/smartcam.example.yml config/
RUN uv sync --frozen --no-dev -p 3.12

ENV SMARTCAM_CONFIG=/app/data/smartcam.yml
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8971/api/cameras >/dev/null || exit 1
CMD ["uv", "run", "--no-sync", "python", "-m", "smartcam"]
