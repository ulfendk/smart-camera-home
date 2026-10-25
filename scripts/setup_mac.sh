#!/usr/bin/env bash
# Set up the native macOS (Apple Silicon) dev environment: tools, go2rtc and the Coral runtime.
# Everything lands in bin/ (gitignored); nothing is installed system-wide except Homebrew formulae.
set -euo pipefail
cd "$(dirname "$0")/.."

GO2RTC_VERSION=v1.9.14
# Google's own 2022 macOS runtime fails on invoke with current LiteRT; this community build matches TF 2.19.
EDGETPU_RELEASE=16.0TF2.19.1-1
EDGETPU_ZIP=libedgetpu-16.0-tf2.19.1-1_MacOS_Silicon

brew install uv ffmpeg libusb
uv sync -p 3.12

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p bin/lib models

curl -fsSL -o "$tmp/go2rtc.zip" "https://github.com/AlexxIT/go2rtc/releases/download/$GO2RTC_VERSION/go2rtc_mac_arm64.zip"
unzip -o -q "$tmp/go2rtc.zip" -d bin && chmod +x bin/go2rtc

curl -fsSL -o "$tmp/edgetpu.zip" "https://github.com/feranick/libedgetpu/releases/download/$EDGETPU_RELEASE/$EDGETPU_ZIP.zip"
unzip -o -q "$tmp/edgetpu.zip" -d "$tmp/edgetpu"
cp "$tmp/edgetpu/$EDGETPU_ZIP/throttled/darwin_arm64/libedgetpu.1.0.dylib" bin/lib/
# The build links MacPorts' libusb; point it at Homebrew's and re-sign after patching.
install_name_tool -change /opt/local/lib/libusb-1.0.0.dylib "$(brew --prefix libusb)/lib/libusb-1.0.0.dylib" bin/lib/libedgetpu.1.0.dylib
codesign -f -s - bin/lib/libedgetpu.1.0.dylib
ln -sf libedgetpu.1.0.dylib bin/lib/libedgetpu.1.dylib

curl -fsSL -o models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite

echo "Done. Check the Coral with: uv run python tools/check_coral.py"
