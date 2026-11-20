import cv2
import numpy as np
import pytest

from parking_rules.signs import ParkingSignDetector, SignDetector, find_plate, render_with_plates

BG = (90, 110, 120)


def scene(kind='parking', plates=(), x=300, y=40, size=60, gray=False, noise=3, seed=0, jpeg=True):
    """640x360 frame with a sign (and its plates) pasted in at (x, y), sign width `size`."""
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 260), (640, 360), (70, 70, 70), -1)
    s = render_with_plates(kind, size, plates, background=BG)
    h = min(s.shape[0], 360 - y)
    img[y:y + h, x:x + size] = s[:h]
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if jpeg:
        img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img


@pytest.mark.parametrize('gray', [False, True])
@pytest.mark.parametrize('size', [30, 40, 80, 150])
def test_parking_sign_found_with_outer_box(gray, size):
    [d] = ParkingSignDetector()(scene(size=size, gray=gray))
    assert d.label == 'parking'
    x1, y1, x2, y2 = d.box
    # the box is the sign's outer edge even if the contour found was the blue square
    assert abs(x1 - 300) <= 2 + size * 0.05 and abs(y1 - 40) <= 2 + size * 0.05
    assert abs((x2 - x1) - size) <= 2 + size * 0.08


def test_round_signs_are_not_parking_signs():
    assert ParkingSignDetector()(scene('no_parking', size=80)) == []


def test_plain_rectangles_are_not_parking_signs():
    rng = np.random.default_rng(3)
    det = ParkingSignDetector()
    found = 0
    for _ in range(30):
        img = np.full((360, 640, 3), BG, np.uint8)
        for _ in range(8):
            x, y = (int(v) for v in rng.integers([0, 0], [560, 280]))
            w = int(rng.integers(15, 80))
            cv2.rectangle(img, (x, y), (x + w, y + w), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
            if rng.random() < 0.5:   # some with a bright vertical bar, a bit like a P stem
                cv2.rectangle(img, (x + w // 3, y + w // 5), (x + w // 2, y + 4 * w // 5), (255, 255, 255), -1)
        for frame in (img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):
            found += len(det(frame))
    assert found == 0


@pytest.mark.parametrize('gray', [False, True])
def test_plate_found_below_parking_sign(gray):
    img = scene(plates=[['8-11 h'], ['max. 3 Std.']], size=80, gray=gray)
    [d] = ParkingSignDetector()(img)
    box = find_plate(img, d.box)
    assert box is not None
    x1, y1, x2, y2 = box
    # two one-line plates at 80 px width: (22 % line + 6 % margin) = 22 px each, plus a 3 px gap -> 48 px
    assert abs(x1 - 300) <= 3 and abs(x2 - 380) <= 3
    assert 80 <= y1 - 40 <= 80 + 6
    assert 42 <= y2 - y1 <= 54


def test_plate_found_below_round_sign():
    img = scene('no_parking', plates=[['Mo-Fr', '7-9 h']], size=80)
    [d] = SignDetector()(img)
    assert find_plate(img, d.box) is not None


@pytest.mark.parametrize('kind', ['parking', 'no_parking'])
def test_no_plate_means_none(kind):
    img = scene(kind, size=80)
    det = ParkingSignDetector() if kind == 'parking' else SignDetector()
    [d] = det(img)
    assert find_plate(img, d.box) is None


def test_blank_white_board_below_is_not_a_plate():
    img = scene(size=80)
    cv2.rectangle(img, (300, 124), (380, 150), (255, 255, 255), -1)   # white, but no text on it
    [d] = ParkingSignDetector()(img)
    assert find_plate(img, d.box) is None


@pytest.mark.parametrize('gray', [False, True])
def test_p_sign_bowl_is_not_reported_as_round_sign(gray):
    from parking_rules.signs import detect_all_signs, render_with_plates
    img = render_with_plates('parking', 300, [['8-11 h'], ['max. 3 Std.']], background=(255, 255, 255))
    if gray:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    labels = [d.label for d in detect_all_signs(img, SignDetector(), ParkingSignDetector())]
    assert labels == ['parking']


def test_round_sign_next_to_p_sign_is_kept():
    from parking_rules.signs import detect_all_signs
    img = scene('parking', size=70, x=60)
    s = scene('no_parking', size=70, x=400)
    img[:, 350:] = s[:, 350:]
    labels = sorted(d.label for d in detect_all_signs(img, SignDetector(), ParkingSignDetector()))
    assert labels == ['no_parking', 'parking']
