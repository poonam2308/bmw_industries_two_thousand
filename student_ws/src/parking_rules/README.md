# Parking Rules

BITT finds things that make a parking spot illegal and refuses to park there.

| Done | Rule | Zone |
|:-:|---|---|
| ✅ | Fire hydrant | 1 m circle around it |
| ✅ | No-parking sign (Zeichen 286: red ring, blue inside, one stripe `\`) | Strip from the sign onward, on its side of the road |
| ✅ | No-stopping sign (Zeichen 283: same with a red X) | Same strip |
| ⬜ | Time-limited signs ("8–11 h, max. 3 Std.") | Needs the extra plate read, e.g. with a vision-language model |
| ⬜ | Driveways, junctions | From the map geometry |

## How it works

```text
camera ─┬▶ detector.py (YOLO "fire hydrant") ─▶ bottom-centre pixel ─▶ ray ∩ ground ────────────┐
        └▶ signs.py (circles + pattern check) ─▶ circle centre + radius ─▶ distance from size ──┤ tf at image time
                                                                                                ▼
          tracker.py  (average sightings in odom; trusted after 3; signs also remember the car's heading and
                       whether the sign was left or right of the car)
       ─▶ zones.py    (hydrant → circle, sign → strip; check_pose(x, y, yaw) → violated zones)
       ─▶ /parking_rules/zones (JSON), /parking_rules/markers (RViz / Foxglove)
```

**Why a strip for signs:** a German no-parking sign does not forbid a circle around itself. It applies *from the sign
onward* in the direction of travel, *only on its side of the road*, until the next junction. So the zone starts at the
sign, runs `sign_zone_length` (3 m, toy scale) in the direction the car was driving when it saw the sign, and reaches
`sign_zone_width` (1.2 m) from the sign into the road. Parking just before the sign or across the road stays legal.

**Why distance from size:** a sign hangs on a pole, so the hydrant trick (where it touches the ground) does not work.
A sign of known diameter D that appears 2r pixels wide is `fx · D / 2r` metres away. **Set `sign_diameter` to the size
of your printed sign.**

| File | What it does |
|---|---|
| [parking_rules/detector.py](parking_rules/detector.py) | YOLO wrapper (hydrants) |
| [parking_rules/signs.py](parking_rules/signs.py) | Sign detector (no training), sign renderer for printing props |
| [parking_rules/geometry.py](parking_rules/geometry.py) | Pixel → ground point; pixel + size → 3D point |
| [parking_rules/tracker.py](parking_rules/tracker.py) | Merges noisy detections into landmarks; majority vote on the sign type |
| [parking_rules/zones.py](parking_rules/zones.py) | `RULES` (label → circle/strip, reason), car footprint, `check_pose()` |
| [parking_rules/mapper_node.py](parking_rules/mapper_node.py) | ROS node that ties it together (`parking_rules_mapper`) |
| [parking_rules/check_spot.py](parking_rules/check_spot.py) | CLI: "may I park at x y yaw?" |
| [parking_rules/offline.py](parking_rules/offline.py) | Detectors on photos / webcam, no ROS |

## On your laptop (no ROS, no car)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest test/                                        # geometry, signs, tracker, zone rules

python3 -m parking_rules.signs no_parking no_parking.png       # printable props (1200 px = 20 cm at 150 dpi)
python3 -m parking_rules.signs no_stopping no_stopping.png

python3 -m parking_rules.offline --source photo.jpg            # writes photo_detected.jpg
python3 -m parking_rules.offline --gray                        # webcam, grayscale like the car's cameras
python3 -m parking_rules.offline --no-hydrants                 # signs only, no ultralytics needed
```

## On the car (dev container)

```bash
pip install --break-system-packages ultralytics                # once
cd student_ws && colcon build --packages-select parking_rules && source install/setup.bash

ros2 run parking_rules parking_rules_mapper --ros-args -p sign_diameter:=0.20
ros2 run parking_rules parking_rules_mapper --ros-args -p image_topic:=/visual_slam/image_0   # simulation

ros2 topic echo /parking_rules/zones
ros2 run parking_rules check_spot 3.0 1.5 0.0                  # x y yaw of base_link in odom → legal / ILLEGAL
ros2 run parking_rules check_spot --here                       # where the car stands now
```

In RViz / Foxglove add `/parking_rules/markers` (red = no parking) and `/parking_rules/debug/compressed` (camera image
with the detections).

Parameters of `parking_rules_mapper` (`--ros-args -p name:=value`):

| Parameter | Default | |
|---|---|---|
| `image_topic` | `/preview/left/compressed` | `/visual_slam/image_0` in the simulation |
| `camera_info_topic` | `/visual_slam/camera_info_0` | Scaled to the image size automatically |
| `camera_frame` | `left_optical` | Used if the image header has no frame |
| `fixed_frame` / `ground_frame` | `odom` / `base_footprint` | |
| `detect_hydrants` / `detect_signs` | `true` / `true` | |
| `sign_diameter` | `0.20` | m, outer diameter of **your** sign |
| `sign_zone_length` / `sign_zone_width` | `3.0` / `1.2` | m |
| `sign_min_radius_px` | `6` | Smaller circles are ignored |
| `rate_hz`, `max_range` | `3`, `8` | |
| `min_conf`, `model` | `0.35`, `yolo11n.pt` | YOLO (hydrants) |
| `gate`, `min_hits` | `0.75`, `3` | Tracker |

### Using it from your parking planner

Subscribe to `/parking_rules/zones` (transient local QoS) and filter your candidate spots before sending the Nav2 goal:

```python
from parking_rules.zones import check_pose, zones_from_json

zones = zones_from_json(msg.data)
legal = [c for c in candidates if not check_pose(c.x, c.y, c.yaw, zones)]
reasons = [z.reason for z in check_pose(x, y, yaw, zones)]   # tell the user why
```

## How good is the sign detector?

Measured on synthetic street frames (sign pasted in, pixel noise, JPEG quality 60; 12 trials each):

| Sign radius in the image | 10 px | 15 px | 25 px | 45 px |
|---|---|---|---|---|
| Colour image | 11/12 | 10–12/12 | 12/12 | 12/12 |
| Grayscale, light noise | 0/12 | 9/12 | 12/12 | 12/12 |
| Grayscale, heavy noise | 0/12 | 3/12 | 12/12 | 12/12 |

False alarms on 60 cluttered frames full of random coloured circles with diagonal stripes: **colour 0, grayscale 28**.

**The stereo cameras are grayscale, and that is the weak spot.** Printed traffic red and blue differ by only ~11 of 255
brightness levels, so in gray a sign is just "a round thing with a slightly different ring and stripe". Any striped
circle can pass. The tracker's 3-sightings rule filters one-off false alarms, but a striped object that is really there
would be confirmed. Ways out, best first:

1. **Check what the real sensor sees.** Print a sign, look at it through `/parking_rules/debug/compressed`. A mono
   sensor may show red and blue very differently from the formula used here; then gray works better than measured.
2. **Use a colour camera.** The rear USB camera is colour (`image_topic:=/back_camera/image/compressed`,
   `camera_frame:=back_camera_optical`; its mount pose in the URDF is a placeholder, so measure it).
3. **Train YOLO on your props.** Photograph the printed signs from the car (grayscale, many distances/angles), label
   them, fine-tune `yolo11n`, and add the classes to `WANTED` in `detector.py`.

## Known limits

- **Signs are only found up to the distance where they are ~10 px in radius** (colour) or ~25 px (gray). With the
  preview's resolution and a 20 cm sign that is only a few metres: drive slowly past them, or print bigger signs.
- **Flat ground** for hydrants: kerbs, slopes and a box cut off at the image bottom give wrong distances.
- **Preview intrinsics:** the node scales `camera_info` to the preview resolution and ignores lens distortion. Check
  whether the preview is the raw or the rectified left image and pick the matching `camera_info_topic`.
- **Sign on the wrong side:** the side is decided by whether the sign was left or right of the car when seen. On a
  narrow road with the car driving in the middle this can flip; the tracker takes the majority.
- **The simulation's street world has neither hydrants nor these signs.** Add models to your local copy of the
  scenario to test the full pipeline in sim.
- **Distances are toy-scale** (BITT ≈ 1.3 m long). Change them in `RULES` in `zones.py` or with the parameters.
