# Rear Buddy: demo day on BITT

Step-by-step guide for showing Rear Buddy live on the BMW toy car. The car's rear camera sees who is behind it.
The rear LEDs and a phone in the rear window react.

```text
rear camera ─▶ BITT Wi-Fi ─▶ your Mac: YOLO + brand check + rules ─▶ rear LEDs (/lights_cmd)
                                                                 └─▶ phone display (web page)
```

All commands run from the package folder, with the virtual environment active:

```bash
cd ~/Documents/myworkspace/bmw_industries_two_thousand/student_ws/src/rear_buddy
source .venv/bin/activate
```

---

## 1. The evening before (at home, WITH internet)

The car's Wi-Fi has **no internet**, so anything that downloads must happen today.

- [ ] **Right branch:** `git status` shows `On branch rear-buddy`.
- [ ] **Tests pass:** `python3 -m pytest -q test/` gives `74 passed`.
- [ ] **Models work offline:** this must print the job pitch, "old is gold" and the hello, with no download:
  ```bash
  HF_HUB_OFFLINE=1 python3 -m rear_buddy.offline --no-show --source ../../../bags/demo_photos
  ```
- [ ] **Backup demo works** (§5): `python3 -m rear_buddy.replay ../../../bags/demo_drive --rate 0`
- [ ] **Your details:** `rear_buddy/profile_local.py` has your name, skills, languages and **the correct LinkedIn link**
  (`QR_URL`). Test: `python3 -m rear_buddy.offline --display --source ../../../bags/demo_photos/Porsche_911_Carrera.jpg`,
  open the address on your phone and **scan the QR code with a second phone**.
- [ ] **Flirt mode:** only use it if the person named in `FLIRT_PROFILE` (in `profile_local.py`) agreed to it.
- [ ] **Props:**
  - printed photos, A4 or bigger, full car visible: one Porsche/Audi/BMW (from `bags/demo_photos/`), one Toyota or
    small car, one bicycle;
  - a phone mount for the car's rear window;
  - laptop charger and phone charger.
- [ ] **Phone:** switch off auto-lock (Settings → Display → Auto-Lock → Never) so it doesn't go dark.
- [ ] **Ask the instructors** for the `BITT` Wi-Fi password, whether the **rear USB camera** is mounted, and whether
  driving `/lights_cmd` from your laptop is OK.

---

## 2. At the car: setup (about 10 minutes before the demo)

1. **Car on:** the car starts its Wi-Fi and pipeline when it boots. Give it a minute.
2. **Mac on the `BITT` Wi-Fi**, then check: `ping 10.42.0.1` gets answers.
3. **Is the rear camera there?** Start Rear Buddy without touching the lights and look at the window:
   ```bash
   python3 -m rear_buddy.bridge --no-leds --show
   ```
   - `connected to ... at ws://10.42.0.1:8765` → the connection works.
   - `first image from /back_camera/image/compressed` and a window with the rear view → the camera works.
   - `no images on /back_camera/image/compressed after 10 s` → no rear camera: see §6.

   Stop with Ctrl-C.
4. **Lights test** with nobody behind the car:
   ```bash
   python3 -m rear_buddy.bridge --show
   ```
   The rear bars show the **red KITT scanner**. If it looks odd, try `--reverse`. If it's too bright or too dark,
   change `--brightness` (default 0.5). Stop with Ctrl-C; **the lights switch off**.
5. **Phone:** connect it to the `BITT` Wi-Fi as well. Mount it in the rear window, facing backwards.

---

## 3. Start the demo

```bash
python3 -m rear_buddy.bridge --display --show --pitch-people
```

`--pitch-people`: a person behind the car gets the job pitch + QR code, because a person can scan it. Without it,
people get a friendly hello and only Porsche/Audi/BMW photos get the pitch. Add `--flirt` for flirt mode (cyclists).

1. Wait for `loading the car brand model (CLIP ViT-L-14) ...`, then `connected to ...` and `first image ...`. Loading
   takes a few seconds.
2. It prints `rear display: open http://10.42.0.xx:8080/ on the phone`. Open **exactly that address** on the phone,
   then **tap the screen** for full screen. The dot in the bottom-right corner turns green when the phone is
   connected.
3. Keep the laptop window visible to the audience if possible. It shows the camera view, the boxes, the message and
   the LED bar.

**Stop:** Ctrl-C in the terminal, or the red ■ in PyCharm (click once). The car's lights switch off.

---

## 4. The demo script (about 5 minutes)

| # | You do / say | The car shows |
|---|---|---|
| 1 | *"Stuck in traffic is boring, for me and for the driver behind me. So BITT entertains them."* Nobody behind the car. | Red **KITT scanner**, display blank |
| 2 | Your **teammate stands behind the car**: *"Who is behind me? Maybe someone who is hiring…"* | 👋💼 gold shine, **job pitch + QR code**. Invite them to scan it. Stay over 20 s and the text changes. |
| 3 | Teammate steps away. You **stand behind the car with your bike**. | 🚲 green wave, cyclist joke. With `--flirt`: ❤️ hearts and the coffee invitation |
| 4 | Bike away. Hold the **Porsche/Audi/BMW photo** behind the car: *"And if a nice car is behind me…"* | 🏎️ gold shine, the job pitch for the car's driver |
| 5 | Hold the **Toyota / small car photo**. | 😊 *"Hi :) Old is gold."* |
| 6 | Everyone steps away. | Back to the scanner after ~8 s |
| 7 | Explain how it works (below), then Ctrl-C: the lights go off. | |

**Tips for the photos:**
- Hold them **close to the camera and bigger than any face** in view: the closest, biggest subject wins. Step aside
  or hold the photo in front of your body, so the camera sees the car and not you.
- Keep the photo still for **2–3 seconds**. A new scene must be seen several times in a row before the display
  changes.

**How it works, in one minute:**
- The car streams its rear camera over Wi-Fi through its Foxglove bridge. No ROS is needed on the laptop.
- On the laptop, **YOLO** finds people, bikes and cars. **CLIP** recognises the brand of the closest car (offline,
  no training).
- **Rules** pick the scene and keep the display calm: a scene must be seen several times, messages stay up at
  least 8 s, and texts rotate in traffic, using the wheel speed from `/wheel_odom`.
- The laptop sends **LED animations** to `/lights_cmd` 10 times per second, and pushes the text to the phone's
  web page.

---

## 5. Backup plan: replay instead of live

If the car, the Wi-Fi or the rear camera doesn't work, show the same software on a recorded drive. No car is needed:

```bash
python3 -m rear_buddy.replay ../../../bags/demo_drive --show --display
```

It's 60 seconds: empty road → person → Porsche (job pitch) → Toyota (old is gold) → VW (traffic-jam joke, because
the recording stands still for over 30 s) → empty road.
- **Phone display:** works on any shared Wi-Fi, or on your phone's hotspot with the Mac connected to it.
- **Laptop only:** without `--display`, the laptop window alone is enough.
- **A second live option:** the webcam, `python3 -m rear_buddy.offline --display`, with the printed photos held up to
  the laptop camera.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ping 10.42.0.1` fails | The Mac isn't on `BITT`, or the car is still booting. Wait a minute, reconnect the Wi-Fi. |
| `retrying every 2 s` | The car's bridge isn't reachable: same as above. Rear Buddy reconnects by itself. |
| `no images on /back_camera/image/compressed` | No rear camera. Ask the instructors, or use the front camera and hold the photos in front of the car: `--topic /preview/left/compressed` |
| Phone can't open the page | Phone on `BITT`? Exactly the printed address (`10.42.0.xx`, not `localhost`)? macOS asked about incoming connections → **Allow**. |
| Phone page frozen (dot red) | Reload the page. |
| A car photo isn't recognised as that brand | Use a bigger, clearer photo of the whole car. Emergency: restart with `--fancy always` (every car gets the pitch) or `--fancy budget` (every car gets "old is gold"). |
| Lights don't react | Did the instructors allow `/lights_cmd`? Does Foxglove's **All off** button work (same topic)? Otherwise demo with the display only: `--no-leds`. |
| Lights stay on after a crash | Restart Rear Buddy and stop it with Ctrl-C (sends "off"), or press **All off** in Foxglove. |
| Light buttons in Foxglove do nothing | Expected while Rear Buddy runs: it overrides `/lights_cmd`. Stop it first. |
| People in the background trigger "Hi there!" | Only people bigger than 5 % of the image count (`--min-person 0.05`). Raise it, e.g. `--min-person 0.1`, so only people close behind the car count. |
| People right behind the car are ignored | Lower it: `--min-person 0.03`. |
| Everything is slow | Lower the load: `--detect-hz 1`. Close other apps. |
| Headlights shouldn't be on | `--front off` |

---

## 7. Afterwards

- Ctrl-C, and check that the car's lights are off.
- If possible, ask the instructors to **record a bag** of the rear camera (`/back_camera/image/compressed`,
  `/wheel_odom`, `/batt_status`). It lets you tune Rear Buddy at home with
  `python3 -m rear_buddy.replay bags/<recording>`.
- Note what didn't work, for example detection at the real camera angle or brand mistakes.
