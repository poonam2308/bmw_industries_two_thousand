"""Try the hydrant and sign detectors without ROS or the car: on photos, a video or your webcam.

    python3 -m parking_rules.offline --source hydrant.jpg          # one photo, writes hydrant_detected.jpg
    python3 -m parking_rules.offline --source photos/              # every image in a folder
    python3 -m parking_rules.offline                               # webcam (q to quit)
    python3 -m parking_rules.offline --gray                        # like the car's mono cameras
    python3 -m parking_rules.offline --no-hydrants                 # signs only (no ultralytics needed)

Use it to check whether your toy/printed props are found at all before driving around with them.
"""
import argparse
import os
import sys

import cv2

from .detector import Detector, draw
from .signs import SignDetector

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def prepare(frame, gray):
    # The car's stereo cameras are monochrome: test with --gray to see what the detector will really get.
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else frame


def run_images(paths, detect, gray):
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f'{path}: cannot read')
            continue
        frame = prepare(frame, gray)
        dets = detect(frame)
        root, ext = os.path.splitext(path)
        out = f'{root}_detected{ext}'
        cv2.imwrite(out, draw(frame, dets))
        found = ', '.join(f'{d.label} {d.conf:.2f}' for d in dets) or 'nothing'
        print(f'{path}: {found} -> {out}')


def run_camera(source, detect, gray):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f'cannot open {source}')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = prepare(frame, gray)
        cv2.imshow('parking_rules', draw(frame, detect(frame)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default='0', help='image, folder, video file or camera index')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--min-conf', type=float, default=0.35)
    p.add_argument('--gray', action='store_true', help='convert to grayscale first (like the car cameras)')
    p.add_argument('--no-hydrants', action='store_true', help='skip YOLO (hydrants)')
    p.add_argument('--no-signs', action='store_true', help='skip the sign detector')
    args = p.parse_args()

    detectors = ([] if args.no_hydrants else [Detector(args.model, args.min_conf)]) + \
                ([] if args.no_signs else [SignDetector()])

    def detector(frame):
        return [d for det in detectors for d in det(frame)]

    if os.path.isdir(args.source):
        paths = sorted(os.path.join(args.source, f) for f in os.listdir(args.source)
                       if f.lower().endswith(IMAGE_EXT) and '_detected' not in f)
        run_images(paths, detector, args.gray)
    elif args.source.lower().endswith(IMAGE_EXT):
        run_images([args.source], detector, args.gray)
    else:
        run_camera(args.source, detector, args.gray)


if __name__ == '__main__':
    main()
