#!/usr/bin/env python3
"""Run Rear Buddy without ROS and without the car: your webcam plays the rear camera.

    python3 -m rear_buddy.offline                         # webcam 0, live window, q to quit
    python3 -m rear_buddy.offline --source car.jpg        # brand check: Porsche/Audi/BMW pitch, Toyota old is gold
    python3 -m rear_buddy.offline --fancy always          # every car gets the job pitch (demo with any car photo)
    python3 -m rear_buddy.offline --source clip.mp4       # a video file
    python3 -m rear_buddy.offline --flirt                 # demo only: the car flirts with cyclists
    python3 -m rear_buddy.offline --source photos/        # every image in a folder, any key for the next one
    python3 -m rear_buddy.offline --source car.jpg --no-show
    python3 -m rear_buddy.offline --display               # also serve the rear display: open the printed address
                                                          # on a phone in the same Wi-Fi

Reactions are printed whenever the display would change; the window shows the rear LED bar on top.
On macOS, allow the terminal to use the camera (System Settings > Privacy & Security > Camera) the first time,
and with --display allow incoming connections for Python when the firewall asks.
"""
import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

from .animations import render
from .detector import Detector
from .display_server import DisplayServer
from .led_preview import bar_image
from .fancy import make_fancy_check
from .messages import QR_SCENES, QR_URL
from .pipeline import analyse, draw
from . import rules
from .rules import Director

IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def with_leds(image, animation, t):
    """The camera view with the rear LED bar (as the car would show it) on top."""
    bar = bar_image(render(animation, t, brightness=1.0), f'rear LEDs: {animation}')
    scale = image.shape[1] / bar.shape[1]
    bar = cv2.resize(bar, (image.shape[1], int(bar.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
    return np.vstack([bar, image])


def print_reaction(r):
    text = r.text.replace('\n', ' / ') or '(blank)'
    print(f'[{r.scene:<10}] leds={r.animation:<10} {text}', flush=True)


def run_images(paths, detector, fancy, show, flirt, on_reaction):
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f'{path}: cannot read', file=sys.stderr)
            continue
        scene, subject, dets = analyse(frame, detector, fancy)
        # A fresh director per image: no confirmation or hold time for single pictures.
        reaction = Director(confirm_frames=1, hold_s=0, flirt=flirt).update(scene)
        print(f'{path.name}: ', end='')
        on_reaction(reaction)
        if show:
            cv2.imshow('rear buddy', with_leds(draw(frame, dets, subject, reaction.text), reaction.animation, 0.5))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


def run_stream(source, detector, fancy, show, flirt, on_reaction):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f'cannot open {source} (camera permission? wrong index or path?)')
    director, text = Director(flirt=flirt), ''
    animation, since = 'scanner', time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scene, subject, dets = analyse(frame, detector, fancy)
        reaction = director.update(scene)
        if reaction:
            on_reaction(reaction)
            text = reaction.text
            if reaction.animation != animation:
                animation, since = reaction.animation, time.monotonic()
        if show:
            image = with_leds(draw(frame, dets, subject, text), animation, time.monotonic() - since)
            cv2.imshow('rear buddy', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default='0', help='webcam index, video file, image file or image folder')
    p.add_argument('--fancy', default='clip', choices=['clip', 'always', 'budget', 'never'],
                   help='car brand check: clip (Porsche/Audi/BMW -> job pitch, Toyota/small -> old is gold) or a fixed answer')
    p.add_argument('--flirt', action='store_true', help='demo only: the car flirts with cyclists')
    p.add_argument('--model', default='yolo11n.pt', help='YOLO weights (downloaded on first use)')
    p.add_argument('--pitch-people', action='store_true', help='people behind the car get the job pitch + QR code')
    p.add_argument('--min-person', type=float, default=rules.MIN_PERSON_AREA,
                   help='smallest person that counts, as a fraction of the image (higher = only people close by)')
    p.add_argument('--conf', type=float, default=0.4, help='minimum detection confidence')
    p.add_argument('--no-show', action='store_true', help='print only, no window')
    p.add_argument('--display', action='store_true', help='serve the rear display page for a phone')
    p.add_argument('--port', type=int, default=8080, help='port of the rear display page')
    args = p.parse_args()
    rules.MIN_PERSON_AREA = args.min_person
    rules.PITCH_PEOPLE = args.pitch_people

    detector = Detector(args.model, args.conf)
    fancy = make_fancy_check(args.fancy)
    on_reaction = print_reaction
    server = None
    if args.display:
        server = DisplayServer(port=args.port, qr_url=QR_URL, qr_scenes=QR_SCENES)
        if not server.qr:
            print('no QR code on the display: pip install segno')
        print('rear display: open ' + ' or '.join(server.urls()) + ' on the phone (same Wi-Fi as this laptop)')

        def on_reaction(r):
            print_reaction(r)
            server.show(r.__dict__)
    src = pathlib.Path(args.source)
    try:
        if src.is_dir():
            run_images(sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_TYPES), detector, fancy,
                       not args.no_show, args.flirt, on_reaction)
        elif src.suffix.lower() in IMAGE_TYPES:
            run_images([src], detector, fancy, not args.no_show, args.flirt, on_reaction)
        else:
            run_stream(args.source, detector, fancy, not args.no_show, args.flirt, on_reaction)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if server:
        server.close()


if __name__ == '__main__':
    main()
