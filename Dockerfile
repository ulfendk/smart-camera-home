# smartcam for Linux, VAAPI decoding on an Intel iGPU. Two variants, built by .github/workflows/image.yml and
# published to GitHub Container Registry:
#   cuda  (default, tag latest): detection on an NVIDIA GPU
#   intel (tag intel):           detection with OpenVINO on the Intel iGPU (Arc), or its CPU
#     docker build --build-arg VARIANT=intel --build-arg BASE=ubuntu:26.04 .
# The config is not in the image: it is read from /app/data/smartcam.yml (the mounted data dir).

# CUDA 12, not 13, so Pascal GPUs (GTX 10-series) still work; NVIDIA has no CUDA 12 image for 26.04.
ARG BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

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
# cuda needs an NVIDIA driver >= 525 on the host.
FROM ${BASE}

ARG VARIANT=cuda
ARG GO2RTC_VERSION=v1.9.14
# Intel's GPU compute runtime (OpenCL, Level Zero) for OpenVINO on the iGPU; Ubuntu's lags behind new chips.
ARG NEO=26.35.39758.10 IGC=2.41.5 IGC_BUILD=22716 GMM=22.10.0
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python LIBVA_DRIVER_NAME=iHD

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg intel-media-va-driver-non-free vainfo ca-certificates curl tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/bin \
    && curl -fsSL -o /app/bin/go2rtc \
        "https://github.com/AlexxIT/go2rtc/releases/download/${GO2RTC_VERSION}/go2rtc_linux_amd64" \
    && chmod +x /app/bin/go2rtc
RUN if [ "$VARIANT" = intel ]; then \
        mkdir /tmp/neo && cd /tmp/neo \
        && for u in \
            intel-graphics-compiler/releases/download/v${IGC}/intel-igc-core-2_${IGC}+${IGC_BUILD}_amd64.deb \
            intel-graphics-compiler/releases/download/v${IGC}/intel-igc-opencl-2_${IGC}+${IGC_BUILD}_amd64.deb \
            compute-runtime/releases/download/${NEO}/intel-opencl-icd_${NEO}-0_amd64.deb \
            compute-runtime/releases/download/${NEO}/libigdgmm12_${GMM}_amd64.deb \
            compute-runtime/releases/download/${NEO}/libze-intel-gpu1_${NEO}-0_amd64.deb; do \
            curl -fsSLO "https://github.com/intel/$u" || exit 1; \
        done \
        && apt-get update && apt-get install -y --no-install-recommends ./*.deb ocl-icd-libopencl1 libze1 \
        && cd / && rm -rf /tmp/neo /var/lib/apt/lists/*; \
    fi

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project -p 3.12
COPY --from=models /models/*.onnx models/
# Face recognition (faces.py): YuNet finds faces (MIT, OpenCV Zoo), AuraFace recognises them (Apache-2.0).
ARG YUNET_REV=3cc26e7f1014a5ee5d74a42acee58bafc9d0a310 AURAFACE_REV=af6d057c9b0ec4071d4c49c80e3539258798b609
RUN curl -fsSL -o models/yunet.onnx \
        "https://huggingface.co/opencv/face_detection_yunet/resolve/${YUNET_REV}/face_detection_yunet_2023mar.onnx" \
    && curl -fsSL -o models/auraface.onnx "https://huggingface.co/fal/AuraFace-v1/resolve/${AURAFACE_REV}/glintr100.onnx"
COPY smartcam smartcam
COPY config/smartcam.example.yml config/
RUN uv sync --frozen --no-dev -p 3.12
# The same onnxruntime module, built with OpenVINO (its libraries included) instead of CUDA.
RUN if [ "$VARIANT" = intel ]; then \
        uv pip uninstall onnxruntime-gpu && uv pip install "onnxruntime-openvino==1.24.1"; \
    fi
# Licence plate models (fast-alpr), fetched now so the camera host needs no internet for them.
RUN uv run --no-sync python -c "from smartcam.plates import reader; reader().load()"

ENV SMARTCAM_CONFIG=/app/data/smartcam.yml
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8971/api/cameras >/dev/null || exit 1
CMD ["uv", "run", "--no-sync", "python", "-m", "smartcam"]
