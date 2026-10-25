# smartcam

A self-hosted NVR for Home Assistant homes. It records your cameras, detects people, and sends a
Home Assistant push notification with a snapshot and clip when something looks suspicious,
especially while nobody is home. At night it can switch on a Dahua camera's own white light when a
person is in view.

**Website:** https://ulfendk.github.io/smart-camera-home/

smartcam was inspired by [Frigate](https://frigate.video), and borrows its good ideas: go2rtc
restreaming, continuous recording with events on top, zones and masks in normalised coordinates,
and MQTT entities for Home Assistant. It was written to fix two problems that kept coming up:

- **The wrong object wins.** A person walking past a parked car, a bike or a stack of tyres turned
  *that object* into the "moving" detection. Here every track carries its own evidence: how far its
  box has travelled, and whether it has stood still long enough to count as stationary. A parked car or
  a lamp never travels, so it can never become the subject of an alert, whatever moves around it
  (`smartcam/tracker.py`).
- **Lamps and sunlight indoors.** Motion is measured relative to each pixel's neighbourhood
  (`smartcam/motion.py`), so smooth light from a lamp or the sun is not motion. A sudden frame-wide
  change is treated as a lighting event. Places where "people" sit without ever moving are remembered
  and need stronger evidence later (`smartcam/static_map.py`).

On one home's labelled Frigate history (305 cases), smartcam alerted on 252/252 real people with no
false person alarms, against Frigate's 237/252 with 6.

## Running with Docker (Linux, NVIDIA GPU)

Every push to `main` publishes `ghcr.io/ulfendk/smart-camera-home:latest` (and `:<commit sha>`). The
image contains the code and the YOLO11 models; your config lives on the host. It uses CUDA 12, so
Pascal GPUs (GTX 10-series) and newer work.

Prerequisites: the NVIDIA driver (>= 525) and the NVIDIA Container Toolkit, so that
`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` works.

1. Create the data folder (`DATA_DIR`, default `/srv/smartcam`) and put your config in it as
   `smartcam.yml`, starting from [`config/smartcam.example.yml`](config/smartcam.example.yml).
   Secrets can go in a `.env` next to it (see [`.env.example`](.env.example)).
2. `docker compose up -d` with this repository's [`docker-compose.yml`](docker-compose.yml), or in
   Portainer: Stacks → Add stack → Repository, URL `https://github.com/ulfendk/smart-camera-home`,
   compose path `docker-compose.yml`, and `DATA_DIR` / `TZ` as stack variables.
3. Check the log for `on CUDAExecutionProvider`, then open `http://<host>:8971`.

Without an Intel iGPU, drop the `/dev/dri` device and set `detect.hwaccel: none`. Run only one
smartcam per set of cameras: two would fight over the lights and the MQTT client id.

## Running on a Mac (development)

```bash
scripts/setup_mac.sh            # Homebrew tools, go2rtc, Coral runtime, venv
cp config/smartcam.example.yml config/smartcam.yml
cp .env.example .env            # camera / MQTT / HA credentials (never committed)
uv run python -m smartcam       # go2rtc + recording + detection + web UI
```

Detection uses YOLO11 exported to ONNX (`models/`, not in git); on Apple Silicon it runs through
CoreML (~33 ms per frame for YOLO11m). To create the models:

```bash
uvx --python 3.12 --from "ultralytics==8.3.*" --with onnx --with onnxslim \
  yolo export model=yolo11m.pt format=onnx imgsz=640 simplify=True
```

Run the tests with `uv run pytest`.

## Using it

- Web UI on port 8971: live view, events with snapshots and clips, a Home/Away toggle, and a
  "false alarm" button on each event.
- go2rtc on port 1984 restreams every camera as `<camera>` (main) and `<camera>_sub`.
- Home Assistant, with `mqtt` enabled: `switch.smartcam_away`, plus a person sensor and a snapshot
  camera per camera. With `ha` enabled: push notifications through `notify.mobile_app_*`, each with
  its own snapshot.

## Configuration

| Section | What it controls |
|---|---|
| `cameras` | streams, indoor/outdoor, zones, masks (normalised 0–1 coordinates, as in Frigate) |
| `rules` | suspicion scoring: loiter time, door zones, night hours, notify threshold |
| `lights` | Dahua white light; `dry_run: true` only logs what it would switch |
| `mqtt` / `ha` | Home Assistant entities and push notifications |
| `detect` | fps, hardware decoding, model |

`${VAR}` and `${VAR:-default}` are filled in from the environment or `.env`.

**Suspicion score.** A confirmed person scores 1. Then:
- away mode adds 3, and a person indoors while away is the maximum, 10
- night adds 1
- loitering in a zone adds 2
- being in a door zone adds 1
- a person on another camera just before adds 1

At 4 or more a push is sent. While away, any person pushes; at home only outdoor events at night do.

## Regression testing against Frigate

If you are coming from Frigate, its database and recordings make a labelled replay set:

```bash
uv run python tools/select_regression_events.py    # picks cases from local/frigate/frigate.db, writes needed.txt
# copy the listed segments into local/frigate/recordings/ (see the script's docstring)
uv run python tools/contact_sheet.py               # thumbnails for labelling -> local/frigate/labels.json
uv run python tools/run_regression.py              # replay through the pipeline, compare with labels and Frigate
```

## Layout

```
smartcam/
  __main__.py      wires everything together
  go2rtc.py        generates the go2rtc config, runs go2rtc
  recorder.py      continuous 10 s segments, retention
  ingest.py        decoded frames (live, or from files for replay)
  motion.py        lighting-robust motion
  detect/yolo.py   YOLO on ONNX Runtime (CUDA / CoreML / CPU)
  tracker.py       tracks + travel/stationary evidence
  static_map.py    learned false-positive spots
  pipeline.py      per-camera motion -> detect -> track -> confirm
  events.py        events, suspicion score, snapshots, clips
  clips.py         clip cutting (stream copy)
  db.py            SQLite
  ha/              MQTT discovery and push notifications
  lights/          Dahua white light and the controller that switches it
  web/             FastAPI + single-page UI
tools/             probe, Coral check, regression tooling
site/              the project website (GitHub Pages)
```

## Roadmap

- Configuration UI in the web app
- Home Assistant add-on / ingress proxy, so the UI opens inside Home Assistant
- Coral USB as a fast first pass in front of the GPU

## License

MIT, see [LICENSE](LICENSE). Not affiliated with Frigate or Home Assistant.
