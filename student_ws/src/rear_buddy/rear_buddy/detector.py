"""Object detection on one frame: what is behind the car.

Uses an off-the-shelf YOLO model (COCO classes). YOLO only knows "car", not the brand: that is decided by a
separate brand check (see fancy.py) on the closest vehicle.
"""
from dataclasses import dataclass

VEHICLES = {'car', 'truck', 'bus'}
BIKES = {'bicycle', 'motorcycle'}
WANTED = VEHICLES | BIKES | {'person'}


@dataclass
class Detection:
    label: str                 # COCO class name: person, bicycle, car, ...
    conf: float                # detector confidence 0..1
    box: tuple                 # (x1, y1, x2, y2) in pixels
    area: float                # box area as a fraction of the image: bigger = closer
    tier: str | None = None    # vehicles only, from the brand check: 'premium', 'budget' or None
    brand: str | None = None   # vehicles only: the recognised brand, None = unknown or not checked


class Detector:
    def __init__(self, model='yolo11n.pt', min_conf=0.4):
        # Imported here so the rules and their tests run without ultralytics installed.
        from ultralytics import YOLO
        self.model = YOLO(model)  # downloads the weights on first use
        self.min_conf = min_conf

    def __call__(self, frame):
        """frame: HxWx3 BGR numpy array (as from OpenCV). Returns the detections we care about."""
        h, w = frame.shape[:2]
        result = self.model(frame, conf=self.min_conf, verbose=False)[0]
        found = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist()):
            label = result.names[int(cls)]
            if label in WANTED:
                x1, y1, x2, y2 = box
                found.append(Detection(label, conf, (x1, y1, x2, y2), (x2 - x1) * (y2 - y1) / (w * h)))
        return found


def closest_vehicle(detections):
    vehicles = [d for d in detections if d.label in VEHICLES]
    return max(vehicles, key=lambda d: d.area, default=None)
