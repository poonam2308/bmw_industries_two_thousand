"""Find fire hydrants in one camera frame.

COCO (what the stock YOLO weights are trained on) already has a `fire hydrant` class, so no training is needed to
start. Add more classes to WANTED when you add more rules (e.g. `stop sign`), or swap in your own weights for
German traffic signs.
"""
from dataclasses import dataclass

WANTED = {'fire hydrant'}


@dataclass
class Detection:
    label: str     # COCO class name
    conf: float    # detector confidence 0..1
    box: tuple     # (x1, y1, x2, y2) in pixels

    @property
    def foot(self):
        """Bottom centre of the box: where the object touches the ground."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


class Detector:
    def __init__(self, model='yolo11n.pt', min_conf=0.35, wanted=WANTED):
        # Imported here so the geometry/zone code and its tests run without ultralytics installed.
        from ultralytics import YOLO
        self.model = YOLO(model)  # downloads the weights on first use
        self.min_conf = min_conf
        self.wanted = set(wanted)

    def __call__(self, frame):
        """frame: HxWx3 BGR numpy array. Returns the detections of the wanted classes."""
        result = self.model(frame, conf=self.min_conf, verbose=False)[0]
        found = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist()):
            label = result.names[int(cls)]
            if label in self.wanted:
                found.append(Detection(label, conf, tuple(box)))
        return found


def draw(frame, detections):
    """Draw the detections onto a copy of the frame (for the debug image)."""
    import cv2
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(out, (int(d.foot[0]), int(d.foot[1])), 4, (0, 255, 255), -1)
        cv2.putText(out, f'{d.label} {d.conf:.2f}', (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out
