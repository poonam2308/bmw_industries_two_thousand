#!/usr/bin/env python3
"""Replay a recorded drive (ROS 2 bag) through Rear Buddy: no ROS, no Docker, no car needed.

    python3 -m rear_buddy.replay bags/drive_1 --list               # what is in the bag
    python3 -m rear_buddy.replay bags/drive_1                      # real time, prints what the car would show
    python3 -m rear_buddy.replay bags/drive_1 --show --display     # + window (camera, boxes, LED bar) + phone display
    python3 -m rear_buddy.replay bags/drive_1 --rate 0             # as fast as possible: just the reactions
    python3 -m rear_buddy.replay bags/drive_1 --rate 2 --start 30  # double speed, from 30 s into the recording

Exactly the code that runs at the car (bridge mode): camera images go through detection, brand check and rules;
wheel speed and battery come from /wheel_odom and /batt_status if they were recorded. The camera may be the car's
CompressedImage or the simulation's raw Image. Times in the output are seconds into the recording. Timing (hold
time, text rotation, "stuck in traffic") follows the recording's clock, also when replaying faster.
"""
import argparse
import collections
import signal
import threading
import time

import cv2

from .bag import bag_topics, read_bag
from .bridge import BATTERY, WHEELS, Runner
from .detector import Detector
from .display_server import DisplayServer
from .fancy import make_fancy_check
from .images import is_image_type
from .messages import QR_SCENES, QR_URL
from .offline import print_reaction, with_leds
from . import rules
from .rules import Director

CAMERA = '/back_camera/image/compressed'


class BagClock:
    """Seconds into the recording; the replay moves it forward."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def pick_camera(topics, wanted=None):
    """The camera topic to use: the requested one, else the rear camera, else any image topic."""
    images = [t for t, ty in topics.items() if is_image_type(ty)]
    if wanted:
        if wanted not in topics:
            raise SystemExit(f'{wanted} is not in the bag. Image topics: {", ".join(images) or "none"}')
        return wanted
    for candidate in [CAMERA, *(t for t in images if 'back' in t), *images]:
        if candidate in images:
            return candidate
    raise SystemExit(f'no camera images in the bag. Topics: {", ".join(sorted(topics)) or "none"}')


def play(bag, runner, clock, camera, rate=1.0, led_hz=10.0, start_s=0.0, on_tick=None, stop=None):
    """Feed the bag to the runner on the bag's clock, ticking it led_hz times per (bag) second.
    rate 1 = real time, 2 = double speed, 0 = as fast as possible. Returns the replayed duration in seconds."""
    period, next_tick, first = 1.0 / led_hz, 0.0, None
    wall_start = time.monotonic()

    def tick(t):
        clock.t = t
        if rate > 0:
            time.sleep(max(0.0, wall_start + t / rate - time.monotonic()))
        runner.tick()
        if on_tick:
            on_tick()

    for t_ns, topic, msg_type, payload in read_bag(bag, {camera, WHEELS, BATTERY}):
        first = t_ns if first is None else first
        t = (t_ns - first) / 1e9 - start_s
        if t < 0:
            continue
        while next_tick <= t:
            tick(next_tick)
            next_tick += period
            if stop is not None and stop.is_set():
                return clock.t
        clock.t = t
        runner.on_message(topic, payload, {'schemaName': msg_type})
    tick(next_tick)  # let the last image be looked at
    return clock.t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bag', help='bag folder (metadata.yaml + .mcap/.db3) or a single .mcap/.db3 file')
    p.add_argument('--list', action='store_true', help='list the topics in the bag and quit')
    p.add_argument('--topic', help=f'camera topic (default: {CAMERA}, else any image topic in the bag)')
    p.add_argument('--rate', type=float, default=1.0, help='playback speed, 0 = as fast as possible')
    p.add_argument('--start', type=float, default=0.0, help='start this many seconds into the recording')
    p.add_argument('--show', action='store_true', help='window with the camera, boxes, message and LED bar')
    p.add_argument('--display', action='store_true', help='serve the rear display page for a phone')
    p.add_argument('--port', type=int, default=8080, help='port of the rear display page')
    p.add_argument('--fancy', default='clip', choices=['clip', 'always', 'budget', 'never'],
                   help='car brand check (as in bridge mode)')
    p.add_argument('--flirt', action='store_true', help='demo only: cyclists get the flirt texts')
    p.add_argument('--pitch-people', action='store_true', help='people behind the car get the job pitch + QR code')
    p.add_argument('--min-person', type=float, default=rules.MIN_PERSON_AREA,
                   help='smallest person that counts, as a fraction of the image (higher = only people close by)')
    p.add_argument('--detect-hz', type=float, default=2.0, help='detections per second of recording')
    p.add_argument('--model', default='yolo11n.pt', help='YOLO weights')
    args = p.parse_args()
    rules.MIN_PERSON_AREA = args.min_person
    rules.PITCH_PEOPLE = args.pitch_people

    topics = bag_topics(args.bag)
    if args.list:
        for topic, msg_type in sorted(topics.items()):
            mark = '  <- camera' if is_image_type(msg_type) else ''
            print(f'{topic:45s} {msg_type}{mark}')
        return
    camera = pick_camera(topics, args.topic)
    print(f'replaying {args.bag}: camera {camera} ({topics[camera]}), '
          f'wheels {"yes" if WHEELS in topics else "no"}, battery {"yes" if BATTERY in topics else "no"}')

    clock = BagClock()
    shown = []

    def on_reaction(r):
        shown.append((clock.t, r.scene))
        print(f'{clock.t:7.1f}s ', end='')
        print_reaction(r)

    display = None
    if args.display:
        display = DisplayServer(port=args.port, qr_url=QR_URL, qr_scenes=QR_SCENES)
        print('rear display: open ' + ' or '.join(display.urls()) + ' on the phone (same Wi-Fi as this laptop)')
    runner = Runner(Detector(args.model), make_fancy_check(args.fancy), Director(flirt=args.flirt, clock=clock),
                    camera, publish=None, display=display, show=args.show, detect_hz=args.detect_hz, clock=clock,
                    on_reaction=on_reaction)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def on_tick():
        if args.show and runner.view is not None:
            cv2.imshow('rear buddy (replay)', with_leds(runner.view, runner.animation, clock.t - runner.anim_since))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop.set()

    try:
        duration = play(args.bag, runner, clock, camera, rate=args.rate, start_s=args.start, on_tick=on_tick,
                        stop=stop)
    finally:
        if display:
            display.close()
        cv2.destroyAllWindows()

    # How long each scene was on the display.
    seconds = collections.Counter()
    for (t, scene), (t_next, _) in zip(shown, shown[1:] + [(duration, None)]):
        seconds[scene] += t_next - t
    print(f'\nreplayed {duration:.1f} s, {runner.images} images, {len(shown)} display changes')
    for scene, s in seconds.most_common():
        print(f'  {scene:14s} {s:6.1f} s on the display')


if __name__ == '__main__':
    main()
