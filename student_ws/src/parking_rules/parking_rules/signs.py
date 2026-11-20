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


# --- Zeichen 314 "Parken": white P on a blue square ------------------------------------------------------------------
#
# Unlike the round signs this one is easy in grayscale: white P on dark blue is high contrast. Candidates are convex
# quadrilaterals (contours); each is compared with a rendered template by normalised correlation.

PARKING = 'parking'
INSET = 0.06   # white border width, fraction of the sign's side


def render_parking(size=256, background=(200, 200, 200)):
    img = np.full((size, size, 3), background, np.uint8)
    s = size
    cv2.rectangle(img, (1, 1), (s - 2, s - 2), WHITE, -1)
    b = int(s * INSET)
    cv2.rectangle(img, (b, b), (s - 1 - b, s - 1 - b), BLUE, -1)
    pt = lambda fx, fy: (int(fx * s), int(fy * s))                                          # noqa: E731
    cv2.rectangle(img, pt(0.33, 0.22), pt(0.45, 0.80), WHITE, -1)                           # stem
    cv2.ellipse(img, pt(0.45, 0.37), (int(0.20 * s), int(0.15 * s)), 0, -90, 90, WHITE, -1)  # bowl
    cv2.ellipse(img, pt(0.45, 0.37), (int(0.09 * s), int(0.065 * s)), 0, -90, 90, BLUE, -1)  # hole
    return img


def _ncc(a, b):
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def _templates():
    g = cv2.cvtColor(render_parking(N * 4), cv2.COLOR_BGR2GRAY)
    full = cv2.resize(g, (N, N), interpolation=cv2.INTER_AREA)
    b = int(N * 4 * INSET)
    inner = cv2.resize(g[b:-b, b:-b], (N, N), interpolation=cv2.INTER_AREA)
    return {'full': full, 'inner': inner}


P_TEMPLATES = _templates()


class ParkingSignDetector:
    def __init__(self, min_side=28, min_score=0.6):   # below ~30 px it is unreliable (and plates unreadable)
        self.min_side, self.min_score = min_side, min_score

    def candidates(self, gray):
        edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150), np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_side ** 2:
                continue
            approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) == 4 and cv2.isContourConvex(approx) and 0.75 < w / h < 1.33 and area > 0.8 * w * h:
                boxes.append((x, y, w, h))
        return boxes

    def classify(self, frame, gray, box):
        x, y, w, h = box
        patch = cv2.resize(gray[y:y + h, x:x + w], (N, N), interpolation=cv2.INTER_AREA)
        # Whole sign AND its right half (the P's bowl) must match: a white bar on a square matches the stem only.
        scores = {k: min(_ncc(patch, t), _ncc(patch[:, N // 2:], t[:, N // 2:])) for k, t in P_TEMPLATES.items()}
        which, score = max(scores.items(), key=lambda kv: kv[1])
        if score < self.min_score:
            return None
        if is_color(frame):   # colour available: the square must also be blue
            hsv = cv2.cvtColor(cv2.resize(frame[y:y + h, x:x + w], (N, N)), cv2.COLOR_BGR2HSV)
            blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 80)
            if blue.mean() < 0.25:
                return None
        if which == 'inner':   # the contour was the blue square: grow the box to the sign's outer edge
            m = INSET / (1 - 2 * INSET)
            x, y, w, h = x - m * w, y - m * h, w * (1 + 2 * m), h * (1 + 2 * m)
        return Detection(PARKING, score, (x, y, x + w, y + h))

    def __call__(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = [d for d in (self.classify(frame, gray, b) for b in self.candidates(gray)) if d]
        return _suppress(found)


def detect_all_signs(frame, round_detector, parking_detector):
    """Round and P signs together. A round-sign hit whose centre lies inside a P sign or inside any sign's extra
    plate is dropped: in grayscale the P's bowl and round letters on plates can look like weak round signs."""
    parking = parking_detector(frame) if parking_detector else []
    round_ = round_detector(frame) if round_detector else []
    excluded = [p.box for p in parking] + [b for b in (find_plate(frame, d.box) for d in parking + round_) if b]

    def inside(d, box):
        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    return [d for d in round_ if not any(inside(d, box) for box in excluded)] + parking


# --- extra plates (Zusatzzeichen) under a sign ---------------------------------------------------------------------

def render_plate(lines, width=256):
    """White plate with a thin black border and black text, one row per line."""
    line_h = int(width * 0.22)
    h = line_h * len(lines) + int(width * 0.06)
    img = np.full((h, width, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, h - 1), (0, 0, 0), max(1, width // 80))
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, width // 70)
    for i, text in enumerate(lines):
        scale = 1.0
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        scale = min(0.85 * width / tw, 0.6 * line_h / th)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        y = int(width * 0.03) + i * line_h + (line_h + th) // 2
        cv2.putText(img, text, ((width - tw) // 2, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img


def render_with_plates(kind, size=256, plates=(), background=(200, 200, 200)):
    """A sign with extra plates hanging below it (each plate a list of text lines)."""
    sign = render_parking(size, background) if kind == PARKING else render_sign(kind, size, background)
    parts = [sign]
    for lines in plates:
        gap = np.full((int(size * 0.04), size, 3), background, np.uint8)
        parts += [gap, render_plate(lines, size)]
    return np.vstack(parts)


def find_plate(frame, sign_box):
    """Bounding box (x1, y1, x2, y2) of the plate(s) directly below a sign, or None if there is none.

    A plate is a bright rectangle about as wide as the sign, starting just below it, with some dark text inside.
    Stacked plates are merged. Plates against a white wall cannot be separated and count as none."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    x1, y1, x2, y2 = sign_box
    w = x2 - x1
    rx1, rx2 = int(max(0, x1 - 0.25 * w)), int(min(W, x2 + 0.25 * w))
    ry1, ry2 = int(min(H, y2)), int(min(H, y2 + 1.8 * w))
    if ry2 - ry1 < 0.12 * w or rx2 - rx1 < 0.6 * w:
        return None
    roi = gray[ry1:ry2, rx1:rx2]
    _, bright = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(0.08 * w) | 1)
    filled = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))   # fill the dark text
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plates = []
    for c in contours:
        x, y, pw, ph = cv2.boundingRect(c)
        cx = rx1 + x + pw / 2
        inner = bright[y + ph // 6:y + ph - ph // 6, x + pw // 10:x + pw - pw // 10]
        text = 1 - inner.mean() / 255 if inner.size else 0     # dark fraction in the interior: the text
        if (0.6 * w <= pw <= 1.4 * w and 0.12 * w <= ph <= 1.2 * w and abs(cx - (x1 + x2) / 2) < 0.3 * w
                and cv2.contourArea(c) > 0.75 * pw * ph and 0.03 < text < 0.5):
            plates.append((rx1 + x, ry1 + y, rx1 + x + pw, ry1 + y + ph))
    plates.sort(key=lambda b: b[1])
    if not plates or plates[0][1] - y2 > 0.35 * w:
        return None
    box = list(plates[0])
    for p in plates[1:]:                       # stacked plates: each starts right below the previous one
        if p[1] - box[3] > 0.15 * w:
            break
        box = [min(box[0], p[0]), box[1], max(box[2], p[2]), p[3]]
    return tuple(box)


def main():
    p = argparse.ArgumentParser(
        description='Render a sign (optionally with extra plates) as a printable PNG (white background).',
        epilog='example: python3 -m parking_rules.signs parking p.png --plate "8-11 h" --plate "max. 3 Std."')
    p.add_argument('kind', choices=LABELS + (PARKING,))
    p.add_argument('out')
    p.add_argument('--size', type=int, default=1200, help='sign width in pixels (1200 px at 150 dpi = 20 cm)')
    p.add_argument('--plate', action='append', default=[],
                   help='an extra plate below the sign; use | for line breaks ("Mo-Fr|8-18 h"); repeat for more')
    a = p.parse_args()
    plates = [t.split('|') for t in a.plate]
    cv2.imwrite(a.out, render_with_plates(a.kind, a.size, plates, background=WHITE))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
