# Parking Rules

BITT finds things that make a parking spot illegal and refuses to park there.

| Done | Rule | Zone |
|:-:|---|---|
| ✅ | Fire hydrant | 1 m circle around it |
| ✅ | No-parking sign (Zeichen 286: red ring, blue inside, one stripe `\`) | Strip from the sign onward, on its side of the road |
| ✅ | No-stopping sign (Zeichen 283: same with a red X) | Same strip |
| ✅ | Parking sign (Zeichen 314: white P on blue) | Same strip; allowed, but see the plate |
| ✅ | Extra plates with times and limits ("Mo-Fr 8-18 h", "max. 2 Std.", "mit Parkschein") under any of these signs | Legality then depends on **when** and **how long** |
| ⬜ | Driveways, junctions | From the map geometry |

## How it works

```text
camera ─┬▶ detector.py (YOLO "fire hydrant") ─▶ bottom-centre pixel ─▶ ray ∩ ground ─────────────┐
        ├▶ signs.py round signs (286/283) ─┐                                                     │
        └▶ signs.py P sign (314) ──────────┴▶ centre + size ─▶ distance from size ───────────────┤ tf at image time
                │                                                                                ▼
                └▶ find_plate ─▶ readers.py (tesseract | claude) ─▶ plate_book.py (vote)
          tracker.py  (average sightings in odom; trusted after 3; signs also remember the car's heading and
                       whether the sign was left or right of the car)
       ─▶ zones.py    (hydrant → circle, sign → strip with the plate's schedule)
       ─▶ /parking_rules/zones (JSON), /parking_rules/markers (RViz / Foxglove)

check_pose(x, y, yaw, zones, when, minutes) ─▶ plates.py decides: legal, or the reason why not
```

**Why a strip for signs:** a German parking sign does not cover a circle around itself. It applies *from the sign
onward* in the direction of travel, *only on its side of the road*, until the next junction. So the zone starts at the
sign, runs `sign_zone_length` (3 m, toy scale) in the direction the car was driving when it saw the sign, and reaches
`sign_zone_width` (1.2 m) from the sign into the road.

**Why distance from size:** a sign hangs on a pole, so the hydrant trick (where it touches the ground) does not work.
A sign of known size D that appears w pixels wide is `fx · D / w` metres away. **Set `sign_diameter` and
`parking_sign_size` to the size of your printed signs.**

### Time-limited signs

1. **Find the plate:** a bright rectangle about as wide as the sign, directly below it, with dark text inside
   (`signs.find_plate`). Stacked plates are merged.
2. **Read it** in a background thread (`readers.py`), a few times from different sightings. The readings vote
   (`plate_book.py`). Until a plate is understood, the zone counts as **illegal** (safe side).
3. **Parse** the text into a schedule (`plates.parse`). It is forgiving about OCR slips (`O`/`0`, `l`/`1`, dashes).

   | Plate text | Understood as |
   |---|---|
   | `8-11 h`, `8 - 11 Uhr`, `von 8 bis 11 Uhr`, `8.30-18 h`, `18-6 h` | time windows (also overnight) |
   | `Mo-Fr`, `Mo, Mi, Fr`, `Sa`, `werktags` (Mo-Sa), `täglich`, `Mon-Fri` | days for the windows that follow |
   | `max. 3 Std.`, `2 Std.`, `1/2 Std.`, `1,5 Std.`, `90 Min.`, `max. 2 h` | maximum parking time |
   | `mit Parkscheibe` / `mit Parkschein`, `Ticket` | parking disc / pay at the machine |

4. **Decide** with the time of day and the planned duration (`plates.violation`). The meaning depends on the sign
   above the plate (as in the StVO):

   | Sign + plate | Meaning |
   |---|---|
   | 286 / 283 + times | The ban applies only during these times |
   | 314 + times + max. duration | During these times you may stay at most that long; outside them parking is free |
   | 314 + times + Parkschein | A ticket is needed during these times (BITT has none, so illegal) |
   | 314 + times only | Parking only during these times |
   | 314 without a plate | Parking allowed |

   Simplification: the time limit is compared with the minutes of your stay that fall inside the windows.

> [!NOTE]
> **The example "8–11 h, max. 3 Std." never bites under the StVO meaning:** the window itself is 3 h long, so no stay
> can spend more than 3 h inside it. If you meant "parking is *only* allowed 8–11 h, for at most 3 h", that is
> a different rule. Change the `parking` branch in `plates.violation` (and its tests in `test/test_plates.py`).
> It is one `if`.

**Time comes from your laptop, not the car** (the car's clock is often wrong, see TROUBLESHOOTING.md), in
`Europe/Berlin` explicitly, because the dev container usually runs in UTC.

### Plate readers

| `plate_reader` | Needs | Good for |
|---|---|---|
| `tesseract` (default) | `apt install tesseract-ocr tesseract-ocr-deu` + `pip install pytesseract` | Offline, works on the car's Wi-Fi. Needs big, sharp text |
| `claude` | `pip install anthropic`, an API key (`ANTHROPIC_API_KEY` or `ant auth login`) and **internet** | Small, blurry or tilted plates. The car's Wi-Fi has no internet: add a second network (phone hotspot, LAN cable). One API call per read; each plate is read at most `max_plate_reads` (5) times |
| `fixed` | `-p plate_text:="Mo-Fr 8-18 h"` | Simulation and tests |
| `none` | | Plates found but never read: signs with a plate always count as illegal |

Readers only transcribe; `plates.parse` interprets. So all readers go through the same tested rules.

| File | What it does |
|---|---|
| [parking_rules/detector.py](parking_rules/detector.py) | YOLO wrapper (hydrants) |
| [parking_rules/signs.py](parking_rules/signs.py) | Round-sign and P-sign detectors, plate finder, renderer for printing props |
| [parking_rules/plates.py](parking_rules/plates.py) | Plate text → schedule → legal at this time for this long? |
| [parking_rules/readers.py](parking_rules/readers.py) | Plate readers: tesseract, Claude, fixed |
| [parking_rules/plate_book.py](parking_rules/plate_book.py) | Per-sign plate readings and votes (thread-safe) |
| [parking_rules/geometry.py](parking_rules/geometry.py) | Pixel → ground point; pixel + size → 3D point |
| [parking_rules/tracker.py](parking_rules/tracker.py) | Merges noisy detections into landmarks; majority vote on the sign type |
| [parking_rules/zones.py](parking_rules/zones.py) | `RULES`, car footprint, `check_pose()` |
| [parking_rules/mapper_node.py](parking_rules/mapper_node.py) | ROS node that ties it together (`parking_rules_mapper`) |
| [parking_rules/check_spot.py](parking_rules/check_spot.py) | CLI: "may I park at x y yaw, now, for 2 h?" |
| [parking_rules/offline.py](parking_rules/offline.py) | Detectors and plate reading on photos / webcam, no ROS |

## On your laptop (no ROS, no car)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest test/                                        # rules, parser, detectors, readers (mocked)

# printable props (1200 px = 20 cm at 150 dpi); | breaks a plate into lines
python3 -m parking_rules.signs no_parking no_parking.png
python3 -m parking_rules.signs no_stopping no_stopping.png
python3 -m parking_rules.signs parking parking.png --plate "8-11 h" --plate "max. 3 Std."
python3 -m parking_rules.signs no_parking no_parking_rush.png --plate "Mo-Fr|7-9 h"

python3 -m parking_rules.offline --source photo.jpg            # detections + plate text + what it means
python3 -m parking_rules.offline --source photo.jpg --plate-reader claude
python3 -m parking_rules.offline --gray                        # webcam, grayscale like the car's cameras
python3 -m parking_rules.offline --no-hydrants                 # signs only, no ultralytics needed
```

## On the car (dev container)

```bash
pip install --break-system-packages ultralytics pytesseract    # once
sudo apt install -y tesseract-ocr tesseract-ocr-deu            # once (offline plate reader)
cd student_ws && colcon build --packages-select parking_rules && source install/setup.bash

ros2 run parking_rules parking_rules_mapper --ros-args -p sign_diameter:=0.20 -p parking_sign_size:=0.20
ros2 run parking_rules parking_rules_mapper --ros-args -p plate_reader:=claude          # laptop has internet
ros2 run parking_rules parking_rules_mapper --ros-args -p image_topic:=/visual_slam/image_0 \
    -p plate_reader:=fixed -p plate_text:="Mo-Fr 8-18 h max. 2 Std."                     # simulation

ros2 topic echo /parking_rules/zones
ros2 run parking_rules check_spot 3.0 1.5 0.0                  # x y yaw of base_link in odom, now, for 1 h
ros2 run parking_rules check_spot --here --for 3h              # where the car stands, for 3 hours
ros2 run parking_rules check_spot --here --at 10:30 --for 90m
ros2 run parking_rules check_spot --here --at "2026-11-21 09:00"
```

In RViz / Foxglove add `/parking_rules/markers` (red = forbidden, amber = parking with conditions, green = parking
area; the labels show the plate's schedule) and `/parking_rules/debug/compressed` (camera image with signs and plates).

Parameters of `parking_rules_mapper` (`--ros-args -p name:=value`):

| Parameter | Default | |
|---|---|---|
| `image_topic` | `/preview/left/compressed` | `/visual_slam/image_0` in the simulation |
| `camera_info_topic` | `/visual_slam/camera_info_0` | Scaled to the image size automatically |
| `camera_frame` | `left_optical` | Used if the image header has no frame |
| `fixed_frame` / `ground_frame` | `odom` / `base_footprint` | |
| `detect_hydrants` / `detect_signs` / `detect_parking_signs` | `true` / `true` / `true` | |
| `sign_diameter` / `parking_sign_size` | `0.20` / `0.20` | m, outer size of **your** printed signs |
| `sign_zone_length` / `sign_zone_width` | `3.0` / `1.2` | m |
| `plate_reader` | `tesseract` | `tesseract`, `claude`, `fixed`, `none` |
| `plate_text` | | Text for `plate_reader:=fixed` |
| `max_plate_reads` | `5` | Reads per plate; stops earlier when two readings agree |
| `sign_min_radius_px` | `6` | Smaller round signs are ignored |
| `rate_hz`, `max_range` | `3`, `8` | |
| `min_conf`, `model` | `0.35`, `yolo11n.pt` | YOLO (hydrants) |
| `gate`, `min_hits` | `0.75`, `3` | Tracker |

### Using it from your parking planner

Subscribe to `/parking_rules/zones` (transient local QoS) and filter your candidate spots before sending the Nav2 goal:

```python
from parking_rules.plates import local_now
from parking_rules.zones import check_pose, zones_from_json

zones = zones_from_json(msg.data)
now = local_now()                                       # laptop clock, Europe/Berlin
legal = [c for c in candidates if not check_pose(c.x, c.y, c.yaw, zones, when=now, minutes=planned_minutes)]
reasons = [z.reason for z in check_pose(x, y, yaw, zones, when=now, minutes=120)]   # tell the user why
```

Ask the user how long they want to park ("Park next to the yellow car for two hours") and pass that as `minutes`.

## How good are the detectors?

Measured on synthetic street frames (sign pasted in, pixel noise, JPEG quality 60):

| | Works from | Colour | Grayscale | False alarms |
|---|---|---|---|---|
| P sign (314) | ~30 px wide | ✅ | ✅ same as colour (white P on blue is high contrast) | 0, also on squares with a white bar |
| Round signs (286/283) | ~10 px radius (colour), ~25 px (gray) | ✅ | ⚠️ weak, see below | colour 0, gray 28 per 60 cluttered frames |
| Plates | found under signs ≥ ~40 px wide (once the sign itself is found) | ✅ | ✅ black on white | blank white boards are rejected |

**Plate *reading* was not measured:** tesseract was not installed where this was built (its tests skip), and the
Claude reader was tested only against a mocked API. Before relying on it, run `offline.py` on photos of your props
taken *from the car*. Expect OCR to need the plate text at least ~15 px tall: with a 20 cm sign that means about 1 m
away. Print the plates large, approach slowly, or use the Claude reader.

**The stereo cameras are grayscale, and that is the weak spot for round signs.** Printed traffic red and blue differ
by only ~11 of 255 brightness levels, so in gray a 286/283 sign is just "a round thing with a slightly different ring
and stripe". Any striped circle can pass. Round hits inside a P sign or on a plate are dropped. The tracker's
3-sightings rule filters one-off false alarms. Ways out, best first:

1. **Check what the real sensor sees.** Print a sign and look at it through `/parking_rules/debug/compressed`.
2. **Use a colour camera.** The rear USB camera is colour (`image_topic:=/back_camera/image/compressed`,
   `camera_frame:=back_camera_optical`; its mount pose in the URDF is a placeholder, so measure it).
3. **Train YOLO on your props.** Photograph the printed signs from the car, label them, fine-tune `yolo11n`, and add
   the classes to `WANTED` in `detector.py`.

## Known limits

- **Plates are only found up close** (sign ≥ ~40 px wide) and **read even closer** (see above).
- **A plate against a white wall** cannot be separated from the wall and counts as no plate.
- **Unread plates are treated as illegal** until a reading is understood. With `plate_reader:=none` (or a reader that
  is not installed), every sign with a plate is illegal.
- **Flat ground** for hydrants: kerbs, slopes and a box cut off at the image bottom give wrong distances.
- **Preview intrinsics:** the node scales `camera_info` to the preview resolution and ignores lens distortion. Check
  whether the preview is the raw or the rectified left image and pick the matching `camera_info_topic`.
- **Sign on the wrong side:** decided by whether the sign was left or right of the car when seen; the tracker takes
  the majority.
- **Not handled:** public holidays, "Anlieger frei" and other exceptions, arrows on plates (where along the road a
  rule starts/ends), end-of-zone signs, residents' parking.
- **The simulation's street world has no hydrants or these signs.** Add models to your local copy of the scenario,
  and use `plate_reader:=fixed` there.
- **Distances are toy-scale** (BITT ≈ 1.3 m long). Change them in `RULES` in `zones.py` or with the parameters.
