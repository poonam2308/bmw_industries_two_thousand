# Rear Buddy

BITT looks through its rear camera and entertains whoever is behind it while we are stuck in traffic.

| Behind the car | Display text | Rear LEDs |
|---|---|---|
| Fancy car | Job pitch: hire me | `gold_shine` |
| Ordinary car | Friendly hello | `breathing` |
| Ordinary car, we stand still > 30 s | Traffic-jam jokes | `sleepy` |
| Cyclist | Cyclist jokes | `green_wave` |
| Cyclist, **flirt mode** (demo only, off by default) | The car asks them out for coffee | `hearts` |
| Pedestrian | Wave at me | `rainbow` |
| Nobody | (blank) | `scanner` |

Texts live in [rear_buddy/messages.py](rear_buddy/messages.py). For the job pitch put your real `PROFILE`, `QR_URL` and
`FLIRT_PROFILE` into `rear_buddy/profile_local.py` (git-ignored, overrides the placeholders in `messages.py`).

## How it works

```text
image ─▶ detector.py (YOLO: person, bicycle, car, ...) ─▶ fancy.py (closest car fancy?) ─▶ rules.classify (scene)
      ─▶ rules.Director (stable: confirm 3 frames, hold 8 s, rotate text every 20 s) ─▶ Reaction(scene, text, animation)
```

- `classify` picks the closest subject. A rider on a bicycle counts as a cyclist, a driver behind the windscreen as
  the car.
- The fancy-car check is a placeholder for now (`--fancy never|always`): YOLO only knows "car". Plug in a car
  make/model classifier or a vision LLM later.

## On your laptop (no ROS, no car): your webcam is the rear camera

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest test/                               # rules, no camera needed
python3 -m rear_buddy.offline                         # webcam, q to quit
python3 -m rear_buddy.offline --fancy always          # hold a sports-car photo up to the camera
python3 -m rear_buddy.offline --flirt                 # demo only
python3 -m rear_buddy.offline --display               # + rear display: open the printed address on your phone
python3 -m rear_buddy.offline --source photos/        # a folder of test images
python3 -m rear_buddy.led_preview                     # all LED animations, one after another
python3 -m rear_buddy.led_preview hearts --save h.png # one animation as a still image (rows = 100 ms)
```

The webcam window shows the rear LED bar on top, playing the current animation.

The first run downloads the YOLO weights (`yolo11n.pt`, about 6 MB).

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

- A real fancy-car check.
