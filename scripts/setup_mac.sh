#!/usr/bin/env bash
# Set up the native macOS (Apple Silicon) dev environment: tools, go2rtc and the YOLO11 models.
# Everything lands in bin/ and models/ (gitignored); nothing is installed system-wide except Homebrew formulae.
set -euo pipefail
cd "$(dirname "$0")/.."

GO2RTC_VERSION=v1.9.14

brew install uv ffmpeg
uv sync -p 3.12

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p bin models

curl -fsSL -o "$tmp/go2rtc.zip" "https://github.com/AlexxIT/go2rtc/releases/download/$GO2RTC_VERSION/go2rtc_mac_arm64.zip"
unzip -o -q "$tmp/go2rtc.zip" -d bin && chmod +x bin/go2rtc

# YOLO11 exported to ONNX (the same models as in the Docker image).
for m in yolo11s yolo11m; do
  if [ ! -f "models/$m.onnx" ]; then
    (cd "$tmp" && uvx --python 3.12 --from "ultralytics==8.3.*" --with onnx --with onnxslim \
      yolo export model="$m.pt" format=onnx imgsz=640 simplify=True)
    mv "$tmp/$m.onnx" models/
  fi
done

# Face recognition models (the same revisions as in the Docker image).
[ -f models/yunet.onnx ] || curl -fsSL -o models/yunet.onnx \
  "https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx"
[ -f models/auraface.onnx ] || curl -fsSL -o models/auraface.onnx \
  "https://huggingface.co/fal/AuraFace-v1/resolve/af6d057c9b0ec4071d4c49c80e3539258798b609/glintr100.onnx"

echo "Done. Next: cp config/smartcam.example.yml config/smartcam.yml && cp .env.example .env"
