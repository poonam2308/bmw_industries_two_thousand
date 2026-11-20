"""Find German no-parking / no-stopping signs without training a network.

    Zeichen 286 "eingeschränktes Haltverbot" (no parking):  red ring, blue inside, one red stripe  \\
    Zeichen 283 "absolutes Haltverbot"       (no stopping): red ring, blue inside, red X           X

Stock YOLO (COCO) does not know these signs, so this is classic computer vision:
  1. HoughCircles finds round things (candidates).
  2. Each candidate is classified by sampling regions of a normalised crop: ring, inside, the two diagonals.
     Colour image: red/blue masks in HSV (robust). Grayscale image: brightness contrast of ring and stripes (weak:
     red and blue have almost the same brightness, see README).
If this is not good enough, train a YOLO model on photos of your props and use it via `Detector(wanted=...)`.

Print your own props:  python3 -m parking_rules.signs no_parking no_parking.png
"""
import argparse

import cv2
import numpy as np

from .detector import Detection

LABELS = ('no_parking', 'no_stopping')

# Sign colours (BGR), close to RAL 3020 traffic red / RAL 5017 traffic blue.
RED = (28, 18, 193)
BLUE = (148, 73, 0)
WHITE = (255, 255, 255)

N = 64            # classifier crop size (px)
RING = (0.80, 0.95)   # red ring, as a fraction of the radius
INSIDE = 0.62     # blue area radius
BAND = 0.08       # half width of the diagonal sampling band (fraction of the radius)
CENTRE = 0.25     # radius around the centre left out of the diagonal bands


def render_sign(kind, size=256, background=(200, 200, 200)):
    """Draw a sign (BGR image, size x size). Used for the tests and for printing props."""
    img = np.full((size, size, 3), background, np.uint8)
    c, r = size // 2, size // 2 - 2
    cv2.circle(img, (c, c), r, WHITE, -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(r * 0.95), RED, -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(r * 0.75), BLUE, -1, cv2.LINE_AA)
    inner = np.zeros((size, size), np.uint8)
    cv2.circle(inner, (c, c), int(r * 0.75), 255, -1)
    stripes = np.zeros_like(inner)
    t = max(2, int(r * 0.17))
    d = int(r * 0.75 * 0.75)
    cv2.line(stripes, (c - d, c - d), (c + d, c + d), 255, t, cv2.LINE_AA)       # \  both signs
    if kind == 'no_stopping':
        cv2.line(stripes, (c + d, c - d), (c - d, c + d), 255, t, cv2.LINE_AA)   # /  only the X
    img[(stripes > 0) & (inner > 0)] = RED
    return img


def _grid():
    u, v = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    rho = np.hypot(u, v)
    inside = rho < INSIDE
    arms = inside & (rho > CENTRE)                       # leave out the centre: both diagonals cross there
    d1 = arms & (np.abs(u - v) / np.sqrt(2) < BAND)      # \  (image y points down)
    d2 = arms & (np.abs(u + v) / np.sqrt(2) < BAND)      # /
    off_diag = inside & (np.abs(u - v) / np.sqrt(2) > 2 * BAND) & (np.abs(u + v) / np.sqrt(2) > 2 * BAND)
    ring = (rho > RING[0]) & (rho < RING[1])
    return ring, off_diag, d1, d2, rho < 1.0


RING_M, BLUE_M, D1_M, D2_M, DISC_M = _grid()


def is_color(frame):
    if frame.ndim == 2 or frame.shape[2] == 1:
        return False
    sample = frame[::8, ::8].astype(np.int16)
    return np.abs(sample[..., 0] - sample[..., 1]).mean() + np.abs(sample[..., 1] - sample[..., 2]).mean() > 4


def crop(frame, cx, cy, r):
    """Square crop around a circle, resized to N x N (pads with edge pixels near the image border)."""
    r = max(r, 2.0)
    x0, y0 = cx - r, cy - r
    m = cv2.getAffineTransform(np.float32([[x0, y0], [x0 + 2 * r, y0], [x0, y0 + 2 * r]]),
                               np.float32([[0, 0], [N - 1, 0], [0, N - 1]]))
    return cv2.warpAffine(frame, m, (N, N), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def classify_color(patch):
    """patch: N x N BGR crop. Returns (label, confidence) or (None, 0)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 10) | (h > 165)) & (s > 80) & (v > 50)
    blue = (h > 95) & (h < 135) & (s > 80) & (v > 40)
    ring, inside = red[RING_M].mean(), blue[BLUE_M].mean()
    d1, d2 = red[D1_M].mean(), red[D2_M].mean()
    if ring < 0.35 or inside < 0.35:
        return None, 0.0
    base = min(ring, inside)
    return _decide(d1, d2, base)


def _decide(d1, d2, base):
    """d1, d2: how much the \\ and / diagonals look like a stripe (0..1). Between the two thresholds = unsure."""
    if d1 > 0.6 and d2 > 0.6:
        return 'no_stopping', float(min(base, d1, d2))
    if d1 > 0.6 and d2 < 0.35:
        return 'no_parking', float(min(base, d1, 1 - d2))
    return None, 0.0


def classify_gray(patch, min_snr=3.0):
    """patch: N x N gray (or BGR) crop.

    Without colour, red and blue differ only slightly in brightness (about 71 vs 60 of 255 for the printed colours;
    on the real sensor it may be more or less). So look for that small difference in the right *places*: the \\
    stripe (present on both signs) must differ from the blue background by more than `min_snr` times the
    background noise; the ring and the / stripe are then tested for the same brightness as the \\ stripe."""
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Smooth first: the red/blue difference is smaller than the pixel noise, an average over 7x7 is not.
    p = cv2.GaussianBlur(patch, (7, 7), 0).astype(np.float32)
    bg = p[BLUE_M]
    m = float(np.median(bg))
    noise = 1.4826 * float(np.median(np.abs(bg - m))) + 1.0   # robust std, +1 for perfectly clean images
    delta = float(p[D1_M].mean()) - m
    if abs(delta) < min_snr * noise:
        return None, 0.0
    like_stripe = (p - m) * np.sign(delta) > abs(delta) / 2      # pixel has the stripe's (= red's) brightness
    ring = like_stripe[RING_M].mean()
    if ring < 0.35:
        return None, 0.0
    base = min(ring, abs(delta) / (2 * min_snr * noise))
    return _decide(like_stripe[D1_M].mean(), like_stripe[D2_M].mean(), base)


class SignDetector:
    def __init__(self, min_radius=6, max_radius=None, hough_param2=28):
        # max_radius None = half the image's short side (a sign filling the frame still counts).
        self.min_radius, self.max_radius, self.param2 = min_radius, max_radius, hough_param2

    def candidates(self, gray):
        circles = cv2.HoughCircles(cv2.medianBlur(gray, 3), cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=2 * self.min_radius, param1=100, param2=self.param2,
                                   minRadius=self.min_radius, maxRadius=self.max_radius or min(gray.shape[:2]) // 2)
        return [] if circles is None else [tuple(c) for c in circles[0]]

    def __call__(self, frame):
        """frame: BGR or gray image. Returns Detections (box = the circle's bounding box)."""
        color = is_color(frame)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = []
        for cx, cy, r in self.candidates(gray):
            label, conf = classify_color(crop(frame, cx, cy, r)) if color else classify_gray(crop(gray, cx, cy, r))
            if label:
                found.append(Detection(label, conf, (cx - r, cy - r, cx + r, cy + r)))
        return _suppress(found)


def _suppress(dets):
    """Keep the most confident of overlapping detections (Hough often finds the rim and the ring)."""
    keep = []
    for d in sorted(dets, key=lambda d: -d.conf):
        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
        if all(np.hypot(cx - (k.box[0] + k.box[2]) / 2, cy - (k.box[1] + k.box[3]) / 2) > (k.box[2] - k.box[0]) / 2
               for k in keep):
            keep.append(d)
    return keep


def main():
    p = argparse.ArgumentParser(description='Render a sign as a printable PNG (white background).')
    p.add_argument('kind', choices=LABELS)
    p.add_argument('out')
    p.add_argument('--size', type=int, default=1200, help='pixels (1200 px at 150 dpi = 20 cm)')
    a = p.parse_args()
    cv2.imwrite(a.out, render_sign(a.kind, a.size, background=WHITE))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
