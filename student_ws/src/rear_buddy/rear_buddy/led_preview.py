#!/usr/bin/env python3
"""Preview the rear LED animations on your laptop, no car needed.

    python3 -m rear_buddy.led_preview                     # cycle through all animations, 4 s each, q to quit
    python3 -m rear_buddy.led_preview hearts              # just one
    python3 -m rear_buddy.led_preview --save strip.png    # still image: every animation over time, no window
"""
import argparse
import time

import cv2
import numpy as np

from .animations import ANIMATIONS, N_PIXELS, render

PX, GAP, BAR_GAP = 18, 4, 22  # pixel size, gap between pixels, gap between the 4 bars


def bar_image(pixels, label=''):
    """The 4 rear bars as they would look on the car, on a dark background."""
    width = 2 * BAR_GAP + N_PIXELS * (PX + GAP) + 3 * BAR_GAP
    top = 24 if label else 0
    img = np.full((top + PX + 2 * GAP, width, 3), 25, np.uint8)
    if label:
        cv2.putText(img, label, (BAR_GAP, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for i, (r, g, b) in enumerate(pixels):
        x = BAR_GAP + i * (PX + GAP) + (i // 8) * BAR_GAP
        cv2.rectangle(img, (x, top + GAP), (x + PX, top + GAP + PX), (b, g, r), -1)  # OpenCV is BGR
    return img


def save_strip(names, path, brightness, seconds=1.5, rate_hz=10):
    """Each animation as rows of frames, top to bottom, sampled like the car node sends them."""
    rows = []
    for name in names:
        rows.append(bar_image(render(name, 0.0, brightness), f'{name}  (one row = {1000 // rate_hz} ms)'))
        rows += [bar_image(render(name, k / rate_hz, brightness)) for k in range(1, int(seconds * rate_hz))]
    cv2.imwrite(path, np.vstack(rows))
    print(f'saved {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('animation', nargs='?', choices=sorted(ANIMATIONS), help='default: all, one after another')
    p.add_argument('--brightness', type=float, default=1.0, help='0..1 (the car node defaults to 0.5)')
    p.add_argument('--seconds', type=float, default=4.0, help='per animation when cycling')
    p.add_argument('--save', metavar='PNG', help='write a still image instead of opening a window')
    args = p.parse_args()
    names = [args.animation] if args.animation else [n for n in ANIMATIONS if n != 'off']

    if args.save:
        save_strip(names, args.save, args.brightness)
        return
    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        name = names[int(t // args.seconds) % len(names)]
        cv2.imshow('rear LEDs', bar_image(render(name, t % args.seconds, args.brightness), name))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
