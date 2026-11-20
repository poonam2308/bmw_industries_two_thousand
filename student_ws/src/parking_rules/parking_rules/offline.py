"""Try the hydrant and sign detectors without ROS or the car: on photos, a video or your webcam.

    python3 -m parking_rules.offline --source hydrant.jpg          # one photo, writes hydrant_detected.jpg
    python3 -m parking_rules.offline --source photos/              # every image in a folder
    python3 -m parking_rules.offline                               # webcam (q to quit)
    python3 -m parking_rules.offline --gray                        # like the car's mono cameras
    python3 -m parking_rules.offline --no-hydrants                 # signs only (no ultralytics needed)
    python3 -m parking_rules.offline --source p.png --plate-reader claude   # read plates with Claude (internet)

For photos, the extra plates under signs are read (default reader: tesseract) and interpreted; the webcam mode only
marks where the plates are (reading is too slow for every frame).

Use it to check whether your toy/printed props are found at all before driving around with them.
"""
import argparse
import os
import sys

import cv2

from .detector import Detection, Detector, draw
from .plates import parse
from .readers import make_reader
from .signs import ParkingSignDetector, SignDetector, detect_all_signs, find_plate

SIGNS = ('no_parking', 'no_stopping', 'parking')


def with_plates(frame, dets):
    """Add a 'plate' detection under every sign that has one. Returns (all detections, [(sign, plate box)])."""
    pairs = [(d, find_plate(frame, d.box)) for d in dets if d.label in SIGNS]
    pairs = [(d, b) for d, b in pairs if b]
    return dets + [Detection('plate', 1.0, b) for _, b in pairs], pairs


def read_plates(frame, pairs, reader):
    for sign, (x1, y1, x2, y2) in pairs:
        plate = frame[int(y1):int(y2), int(x1):int(x2)]
        both = frame[int(sign.box[1]):int(y2), int(min(sign.box[0], x1)):int(max(sign.box[2], x2))]
        text = reader(both, plate) if reader else None
        schedule = parse(text) if text else None
        shown = (text or '').replace('\n', ' | ')
        result = schedule.describe() if schedule else ('not understood' if reader else 'not read (no reader)')
        print(f'    {sign.label} plate: "{shown}" -> {result}')


IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def prepare(frame, gray):
    # The car's stereo cameras are monochrome: test with --gray to see what the detector will really get.
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else frame


def run_images(paths, detect, gray, reader=None):
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f'{path}: cannot read')
            continue
        frame = prepare(frame, gray)
        dets, pairs = with_plates(frame, detect(frame))
        root, ext = os.path.splitext(path)
        out = f'{root}_detected{ext}'
        cv2.imwrite(out, draw(frame, dets))
        found = ', '.join(f'{d.label} {d.conf:.2f}' for d in dets) or 'nothing'
        print(f'{path}: {found} -> {out}')
        read_plates(frame, pairs, reader)


def run_camera(source, detect, gray):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f'cannot open {source}')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = prepare(frame, gray)
        cv2.imshow('parking_rules', draw(frame, with_plates(frame, detect(frame))[0]))
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
    p.add_argument('--no-signs', action='store_true', help='skip the sign detectors')
    p.add_argument('--plate-reader', default='tesseract', help='tesseract | claude | none')
    args = p.parse_args()

    hydrants = None if args.no_hydrants else Detector(args.model, args.min_conf)
    signs = (None, None) if args.no_signs else (SignDetector(), ParkingSignDetector())
    try:
        reader = make_reader(args.plate_reader)
    except (RuntimeError, ValueError) as e:
        print(f'note: {e}; plates will be found but not read')
        reader = None

    def detector(frame):
        return (hydrants(frame) if hydrants else []) + detect_all_signs(frame, *signs)

    if os.path.isdir(args.source):
        paths = sorted(os.path.join(args.source, f) for f in os.listdir(args.source)
                       if f.lower().endswith(IMAGE_EXT) and '_detected' not in f)
        run_images(paths, detector, args.gray, reader)
    elif args.source.lower().endswith(IMAGE_EXT):
        run_images([args.source], detector, args.gray, reader)
    else:
        run_camera(args.source, detector, args.gray)


if __name__ == '__main__':
    main()
