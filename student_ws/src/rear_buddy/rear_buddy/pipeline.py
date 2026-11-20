"""One frame through the whole chain: detect, check the brand of the closest car, classify. Shared by the
offline runner (laptop webcam / files) and the ROS node (car's rear camera)."""
import cv2

from .detector import closest_vehicle
from .rules import classify


def analyse(frame, detector, car_check):
    detections = detector(frame)
    car = closest_vehicle(detections)
    if car is not None:
        car.tier, car.brand = car_check(frame, car)
    scene, subject = classify(detections)
    return scene, subject, detections


TIER_COLOURS = {'premium': (0, 215, 255), 'budget': (0, 140, 255)}  # BGR: gold, orange
COLOURS = {'person': (0, 200, 255), 'bicycle': (80, 220, 80), 'motorcycle': (80, 220, 80)}


def draw(frame, detections, subject, text):
    """Boxes (the chosen subject thicker) and the message as a banner at the bottom: a preview of the display."""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        colour = TIER_COLOURS.get(d.tier) or COLOURS.get(d.label, (255, 160, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 4 if d is subject else 1)
        brand = f' {d.brand} ({d.tier or "other"})' if d.brand else (f' ({d.tier})' if d.tier else '')
        label = f'{d.label}{brand} {d.conf:.2f}'
        cv2.putText(out, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    lines = [line for line in text.split('\n') if line]
    if lines:
        h, w = out.shape[:2]
        top = h - 12 - 26 * len(lines)
        cv2.rectangle(out, (0, top - 8), (w, h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (10, top + 18 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return out
