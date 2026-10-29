# SmartCam

**The camera alarm for Home Assistant.** SmartCam records your cameras, detects people, and sends a
Home Assistant push notification with a snapshot and clip when something looks suspicious, especially
while nobody is home. Armed, it turns a serious event into an alarm: the cameras' red/blue warning
lights flash. At night a camera's Home Assistant lights go to full while a person is there, the lights
replay a real evening while you're away, and cameras whose stream has died are restarted. It runs on
its own machine and lives in the Home Assistant sidebar.

**Website:** https://ulfendk.github.io/smart-camera-home/

SmartCam was inspired by [Frigate](https://frigate.video), and borrows its good ideas: go2rtc
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

On one home's labelled Frigate history (305 cases), SmartCam alerted on 252/252 real people with no
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
SmartCam per set of cameras: two would fight over the lights and the MQTT client id.

## Running with Docker (Linux, Intel without NVIDIA)

For an Intel machine without an NVIDIA GPU, such as a Core Ultra mini PC, use the `intel` image
(`ghcr.io/ulfendk/smart-camera-home:intel`, also `:intel-<commit sha>`). It runs detection with OpenVINO on
the integrated graphics and decodes video there with VAAPI. It is built on Ubuntu 26.04 with Intel's current
GPU compute runtime, for recent chips (Arc, Core Ultra); older iGPUs such as UHD 630 are no longer supported
by that runtime and fall back to `openvino_device: CPU`.

1. Data folder and config as above, with
   ```yaml
   detect:
     hwaccel: vaapi
     openvino_device: GPU    # or CPU; NPU is not in the image yet
   ```
2. `docker compose -f docker-compose.intel.yml up -d` with [`docker-compose.intel.yml`](docker-compose.intel.yml)
   (it passes `/dev/dri` through; no NVIDIA toolkit needed).
3. Check the log for `on OpenVINOExecutionProvider (GPU)`, then open `http://<host>:8971`.

## Running on a Mac (development)

```bash
scripts/setup_mac.sh            # Homebrew tools, go2rtc, venv, YOLO11 models
cp config/smartcam.example.yml config/smartcam.yml
cp .env.example .env            # camera / MQTT / HA credentials (never committed)
uv run python -m SmartCam       # go2rtc + recording + detection + web UI
```

Detection uses YOLO11 exported to ONNX (`models/`, not in git, created by the setup script); on Apple
Silicon it runs through CoreML (~33 ms per frame for YOLO11m).

Run the tests with `uv run pytest`.

## In the Home Assistant sidebar

The repository is also a Home Assistant add-on repository. Add
`https://github.com/ulfendk/smart-camera-home` under Settings → Add-ons → Add-on Store → ⋮ →
Repositories, install **SmartCam**, and set `server` to your SmartCam address (for example
`http://192.168.1.10:8971`). The UI then opens in the sidebar behind Home Assistant's login, live view
included, also remotely. See [smartcam_proxy/DOCS.md](smartcam_proxy/DOCS.md).

## Using it

- Web UI on port 8971: live view (and a wall view: every camera on one screen, outlining and enlarging the one that
  sees a person; open it directly at `/#wall`, or as one video put together by SmartCam at `/#wall-feed`, also
  as the RTSP stream `wall` for a TV or VLC and as `camera.smartcam_wall` in Home Assistant), events with snapshots and clips (filter by type, camera and time; download buttons), a Home/Away toggle, a
  "false alarm" button on each event (SmartCam remembers the spot: a "person" there then needs a confident
  detection that really walks), and Settings: every option, zones and masks drawn on a live
  snapshot, and a restart to apply. Saving keeps the file's comments and `${VAR}` references, and
  literal passwords are never sent to the browser.
- go2rtc on port 1984 restreams every camera as `<camera>` (main) and `<camera>_sub`.
- Live cameras in Home Assistant: switch on "Share streams on the LAN" and set SmartCam's address under
  Settings → Home Assistant, then press "Add live cameras to Home Assistant" (`camera.smartcam_<name>`).
- Home Assistant, with `mqtt` enabled: `switch.smartcam_away`, plus a person sensor and a snapshot
  camera per camera. With `ha` enabled: push notifications through `notify.mobile_app_*`, each with
  its own snapshot.
- Existing alarm setup: with `ha.alarm_entity` (e.g. an `input_boolean.alarm` that a keypad and scripts
  set), SmartCam arms and disarms with it, and arming SmartCam sets it, so keypads, locks and presence
  simulation stay in step. `ha.away_entity` does the same for away mode.
- Presence simulation while armed: replays a real evening from when you were home (same weekday or
  weekend, all lights together), warped to today's sunrise and sunset with bedtime kept, learned from Home
  Assistant's light history into SmartCam's own library (so it outlives the recorder). Hand-switched lights
  are left alone; on disarm only its own lights are switched off. Settings → Presence simulation previews
  today's plan.
- Working from home: `switch.smartcam_wfh` in Home Assistant (or WFH at the top of the UI) pushes every person an
  outdoor camera sees, day or night, whatever the score; person masks keep the street out. Only people arriving
  after it is switched on count, and going away switches it off.
- Privacy at home: cameras with `record_at_home: false` (e.g. indoors) record nothing and store no events while
  you're home; they record again when armed away. Every camera still reports room occupancy,
  `binary_sensor.smartcam_<camera>_occupied`, for Home Assistant's lights (people sitting still included,
  nothing stored).
- Camera clocks: Dahua clocks are set to this machine's time when they drift (every 6 hours), ONVIF cameras
  (`onvif_port`) get this machine's time zone, so the time in the picture is right.
- Doorbell picture: when the bell's Home Assistant entity turns on, the chosen cameras (e.g. the front door
  outside and the hall inside) are joined into one picture, `camera.smartcam_doorbell`, kept as a "doorbell"
  event and pushed (time-sensitive).
- Lights on people after dark: a camera's `person_lights` (Home Assistant lights) go to full brightness while a
  person is there after dark (sun below the horizon, or a lux sensor), and return to exactly how they were two
  minutes after the person has gone.
- Vehicle speed: draw Speed line A and B across the road in a camera's zone editor and enter the distance
  between them. Each passing vehicle becomes an event with its speed, a full-resolution best shot, a
  close-up and its licence plate when it is readable (fast-alpr, on the CPU); `sensor.smartcam_<camera>_vehicle_speed` in Home Assistant, and a push
  from a chosen speed.
- Alarm mode: `alarm_control_panel.smartcam_alarm` in Home Assistant. Arming switches on away mode;
  a serious event while armed (by default a person indoors, or loitering at a door at night) flashes
  the Dahua red/blue warning lights for 15 minutes, then stays quiet until disarmed and armed again.
  Arming has an exit delay (60 s) and a trigger an entry delay (30 s, "pending") in which disarming cancels it;
  only activity that starts after arming counts. The alarm push can go to other services than the everyday alerts
  (`ha.alarm_notify`, e.g. a household notify group). **Armed home** (e.g. at bedtime) ignores indoors and pushes
  only very suspicious activity outside (loitering, at the door, camera to camera), without setting anything
  off; a person on a chosen camera in the morning (e.g. the stairs after 05:00) or a set time disarms it.
  Every alarm push has **Confirm alarm** and **False alarm** buttons. The answer replaces the push on every phone (with
  who answered), is fired to Home Assistant as `smartcam_alarm_confirmed` / `smartcam_alarm_false` for your own
  escalation, and shows in `sensor.smartcam_alarm_incident`; a false alarm switches the warning lights off and teaches
  the camera spot. With `escalation.enabled`, a confirmed alarm turns every light (or the ones you choose) to full and
  plays a siren on your speakers (SmartCam's own sound, through `media_player.play_media`); the lights are saved as a
  Home Assistant scene and the Sonos speakers snapshotted first, and "Stop siren and lights" on the next push,
  disarming or the time running out puts both back.
- Smoke alarm: with `smoke.entities` (Home Assistant smoke sensors), a sensor going off pushes the indoor cameras as one
  picture, critical, to the alarm's notify services, with the same buttons, armed or not.
- People: everyone in Home Assistant is on SmartCam's people list as family; add others in Settings → People (friends,
  neighbours, grandparents). With `faces.enabled` and `recognize: true` on cameras where faces are big enough, SmartCam
  looks for a face in full-resolution stills of each person: YuNet ([MIT](https://github.com/opencv/opencv_zoo)) finds
  it and AuraFace ([Apache-2.0](https://huggingface.co/fal/AuraFace-v1), an ArcFace ResNet100) recognises it, on the
  same GPU as detection. Faces are learned by naming an event ("Who is this?"). A name is carried to the next camera
  when the clothes' colours match and nobody else fits ("Farmor? at forhave"), also backwards. Pushes say who it is,
  Home Assistant gets a `smartcam_person` event, family is not pushed while working from home or armed home, and a
  known person never stops an alarm (they are named in it). Faces stay on this machine; removing a person removes
  theirs, and unnamed faces go with their events.

## Configuration

| Section | What it controls |
|---|---|
| `cameras` | streams, indoor/outdoor, zones, masks (normalised 0–1 coordinates, as in Frigate) |
| `rules` | suspicion scoring: loiter time, door zones, night hours, notify threshold |
| `lights` | Dahua white light; `dry_run: true` only logs what it would switch |
| `mqtt` / `ha` | Home Assistant entities and push notifications |
| `detect` | fps, hardware decoding, model |
| `alarm` | alarm mode: trigger score, exit and entry delays, red/blue warning lights |
| `smoke` | smoke sensors that push a picture from inside |
| `escalation` | a confirmed alarm: lights to full, siren on the speakers |
| `faces` | person recognition: on/off, similarity threshold, how long a push waits for a name |
| `record` | what stays on disk: a rolling buffer of everything (`continuous_hours`), event clips and snapshots (`event_days`), and a free-space guard (`min_free_gb`) |

`${VAR}` and `${VAR:-default}` are filled in from the environment or `.env`.

**Suspicion score.** A confirmed person scores 1. Then:
- away mode adds 3, and a person indoors while away is the maximum, 10
- night adds 1
- loitering in a zone adds 2
- being in a door zone adds 1
- a person on another camera just before adds 1

At 4 or more a push is sent. While away, any person pushes; at home only outdoor events at night do,
or any outdoor person while working from home.

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
  faces.py         who it is: faces, and names carried from camera to camera
  people.py        the people list and learned faces
  smoke.py         smoke alarm pictures
  escalation.py    a confirmed alarm: lights and siren
  clips.py         clip cutting (stream copy)
  db.py            SQLite
  ha/              MQTT discovery and push notifications
  lights/          Dahua white light and the controller that switches it
  web/             FastAPI + single-page UI
tools/             camera probe, regression tooling
site/              the project website (GitHub Pages)
smartcam_proxy/    Home Assistant add-on: the UI in the sidebar through ingress
```

## Roadmap

- Licence plates for vehicles arriving on a driveway camera, where the plate faces the camera (speed
  cameras usually see cars from the side)

## Support

SmartCam is free and MIT licensed. If it's useful to you, you can support it through
[GitHub Sponsors](https://github.com/sponsors/ulfendk) or [Buy Me a Coffee](https://buymeacoffee.com/ulfendk).

## License

MIT, see [LICENSE](LICENSE). Not affiliated with Frigate or Home Assistant.
