# Rear Buddy

BITT looks through its rear camera and entertains whoever is behind it while we are stuck in traffic.

| Behind the car | Display text | Rear LEDs |
|---|---|---|
| Porsche, Audi, BMW | Job pitch: hire me (+ LinkedIn QR code) | `gold_shine` |
| Toyota, small car | Hi :) Old is gold. | `breathing` |
| Other car (or brand unsure) | Friendly hello | `breathing` |
| Ordinary car, we stand still > 30 s | Traffic-jam jokes | `sleepy` |
| Cyclist | Cyclist jokes | `green_wave` |
| Cyclist, **flirt mode** (demo only, off by default) | The car asks them out for coffee | `hearts` |
| Pedestrian | Wave at me | `rainbow` |
| Nobody | (blank) | `scanner` |

Texts live in [rear_buddy/messages.py](rear_buddy/messages.py). For the job pitch put your real `PROFILE`, `QR_URL` and
`FLIRT_PROFILE` into `rear_buddy/profile_local.py` (git-ignored, overrides the placeholders in `messages.py`).

## How it works

**Car brands**: the closest car is cropped and recognised with CLIP (open_clip ViT-L-14), zero-shot, offline once the
weights (~1.7 GB) are downloaded on the first run; download them at home, the car's Wi-Fi has no internet. Brand lists:
`CAR_BRANDS` and `BUDGET_DESCRIPTIONS` in [messages.py](rear_buddy/messages.py); when the model is unsure (< 50 %) the
car gets the normal hello. `--fancy always|budget|never` replaces the check with a fixed answer for tests and demos.
On 8 test photos: Porsche, Audi, BMW, 2 Toyotas, Mini and VW right; a BMW police car came out unsure (normal hello).

```text
image ─▶ detector.py (YOLO: person, bicycle, car, ...) ─▶ fancy.py (closest car: brand?) ─▶ rules.classify (scene)
      ─▶ rules.Director (stable: confirm 3 frames, hold 8 s, rotate text every 20 s) ─▶ Reaction(scene, text, animation)
```

- `classify` picks the closest subject. A rider on a bicycle counts as a cyclist, a driver behind the windscreen as
  the car.
- YOLO only knows "car": the brand comes from the CLIP check in `fancy.py` (see "Car brands" above).

## On your laptop (no ROS, no car): your webcam is the rear camera

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest test/                               # rules, no camera needed
python3 -m rear_buddy.offline                         # webcam, q to quit
python3 -m rear_buddy.offline --source car.jpg        # brand check on a photo (Porsche/Audi/BMW, Toyota, ...)
python3 -m rear_buddy.offline --fancy always          # every car gets the job pitch
python3 -m rear_buddy.offline --flirt                 # demo only
python3 -m rear_buddy.offline --display               # + rear display: open the printed address on your phone
python3 -m rear_buddy.offline --source photos/        # a folder of test images
python3 -m rear_buddy.led_preview                     # all LED animations, one after another
python3 -m rear_buddy.led_preview hearts --save h.png # one animation as a still image (rows = 100 ms)
```

The webcam window shows the rear LED bar on top, playing the current animation.

The first run downloads the YOLO weights (`yolo11n.pt`, about 6 MB).

## With the car, from your laptop (no ROS, no Docker): bridge mode

The car runs a Foxglove bridge (`ws://10.42.0.1:8765`). Bridge mode talks to it directly: rear camera, `/wheel_odom`
and `/batt_status` in, `/lights_cmd` out, and the phone display served from the laptop.

```bash
# laptop on the BITT Wi-Fi
python3 -m rear_buddy.bridge --display --show          # camera window + phone display, LEDs on the car
python3 -m rear_buddy.bridge --display --fancy always   # demo the job pitch with a car photo
python3 -m rear_buddy.bridge --display --flirt          # demo only
python3 -m rear_buddy.bridge --no-leds --show           # watch only, leave the car's lights alone
python3 -m rear_buddy.bridge --topic /preview/left/compressed --show   # front camera, if the rear one is missing
```

- Ctrl-C (or closing the terminal) switches the car's lights off and disconnects.
- While it runs it overrides the light buttons in Foxglove (`/lights_cmd` is "last message wins").
- It reconnects by itself after Wi-Fi drops. After 10 s without camera images it tells you so.
- How: [foxglove_client.py](rear_buddy/foxglove_client.py) speaks the Foxglove WebSocket protocol,
  [cdr.py](rear_buddy/cdr.py) (de)serialises the ROS messages, [bridge.py](rear_buddy/bridge.py) runs the same
  detection, rules, LEDs and display as everywhere else.

## Replay a recorded drive (no ROS, no Docker, no car)

Recorded bags (rosbag2, `.mcap` or `.db3`) from the car or the simulation play through exactly the code that runs at
the car: detection, brand check, rules, LED bar and phone display, on the recording's clock.

```bash
python3 -m rear_buddy.replay bags/drive_1 --list                # what is in the bag
python3 -m rear_buddy.replay bags/drive_1 --show --display      # real time, window + phone display
python3 -m rear_buddy.replay bags/drive_1 --rate 0              # as fast as possible: the list of reactions
python3 -m rear_buddy.replay bags/drive_1 --rate 2 --start 30   # double speed from 30 s in
```

- The camera may be the car's `CompressedImage` or the simulation's raw `Image`; `--topic` picks another camera.
- `/wheel_odom` and `/batt_status` are used when recorded (standing still > 30 s behind a car -> traffic-jam jokes).
- At the end it prints how long each scene was on the display.

## On the car (dev container)

```bash
pip install --break-system-packages ultralytics segno # once
cd student_ws && colcon build --packages-select rear_buddy && source install/setup.bash
ros2 launch rear_buddy rear_buddy.launch.py           # camera + LEDs + display; fancy:=always flirt:=true brightness:=0.3
ros2 topic echo /rear_buddy/reaction

# or one node at a time
ros2 run rear_buddy rear_buddy_node                    # -p fancy:=always  -p flirt:=true  -p rate_hz:=2.0
ros2 run rear_buddy rear_buddy_leds --ros-args -p animation:=rainbow     # try one animation on the real lights
```

The node reads `/back_camera/image/compressed` (the rear USB camera must be connected), `/wheel_odom` (how long we
stand still) and `/batt_status` (battery texts). It publishes `/rear_buddy/reaction` (JSON) whenever the display should
change.

`rear_buddy_leds` follows `/rear_buddy/reaction` and streams the animation to `/lights_cmd` at 10 Hz
(parameters: `brightness` 0.5, `rate_hz` 10, `front` white|off for the headlights, `reverse` if sweeps run the wrong
way). Animations live in [rear_buddy/animations.py](rear_buddy/animations.py): a function `(pixel, time) -> rgb`.

- `/lights_cmd` is "last message wins": while the LED node runs, the light buttons in Foxglove / `controls.py`
  (indicators, hazard) are overridden. Stop the node to use them. Ctrl-C switches all lights off.
- The firmware still adds its own brake light on top.
- The simulation ignores `/lights_cmd`: watch `/lights_state` on the car, or use `led_preview` on your laptop.
- Not tried on the car yet: check the pixel direction and whether 10 Hz over Wi-Fi looks smooth (`rate_hz`).

## Rear display (phone now, small screen later)

A tiny web server on the laptop ([display_server.py](rear_buddy/display_server.py), standard library only) serves
[display.html](rear_buddy/display.html): the current message full-screen, auto-sized, with emojis and a moving
background per scene (gold shine, floating hearts, KITT scanner when nobody is there, ...). New messages are pushed
instantly (Server-Sent Events) and the page reconnects by itself; the dot bottom right is green while connected.

1. Start it: `python3 -m rear_buddy.offline --display` on the laptop, or `rear_buddy_display` / the launch file with
   ROS. It prints the address, e.g. `http://192.168.1.23:8080/`.
2. Phone on the **same Wi-Fi** as the laptop (at home your Wi-Fi, at the car `BITT`), open that address, tap for full
   screen. iPhone: Share > Add to Home Screen opens it without browser bars.
3. Switch off auto-lock on the phone, or it goes dark.

The job pitch shows a **QR code to LinkedIn** (`QR_URL` / `QR_SCENES` in [messages.py](rear_buddy/messages.py),
generated on the laptop with `segno`, so it works without internet). Without segno installed the display shows no QR
code; on the car `-p qr_url:=''` switches it off.

The first time, macOS asks whether Python may accept incoming connections: allow it. The page has no internet
dependencies (the car's Wi-Fi has none). A small screen later: any device with a browser opening the same address.

## Next steps

- Check the brand recognition with the real rear camera (angle, distance, light) and tune `min_conf`.
