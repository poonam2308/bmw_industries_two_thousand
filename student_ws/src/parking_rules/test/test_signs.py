import cv2
import numpy as np
import pytest

from parking_rules.geometry import pixel_to_point_by_size
from parking_rules.signs import SignDetector, classify_color, classify_gray, crop, render_sign

INTR = (500.0, 500.0, 320.0, 180.0)


def scene(kind, cx=400, cy=150, r=30, gray=False, noise=0, seed=0):
    """A 640x360 street-ish frame with one sign pasted in."""
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), (90, 110, 120), np.uint8)
    cv2.rectangle(img, (0, 220), (640, 360), (70, 70, 70), -1)   # road
    if kind:
        s = render_sign(kind, 2 * r + 4, background=(90, 110, 120))
        img[cy - r - 2:cy + r + 2, cx - r - 2:cx + r + 2] = s
    if noise:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


@pytest.mark.parametrize('kind', ['no_parking', 'no_stopping'])
def test_classify_rendered_sign_color(kind):
    img = render_sign(kind, 128)
    assert classify_color(crop(img, 64, 64, 62))[0] == kind


@pytest.mark.parametrize('kind', ['no_parking', 'no_stopping'])
def test_classify_rendered_sign_gray(kind):
    img = cv2.cvtColor(render_sign(kind, 128), cv2.COLOR_BGR2GRAY)
    assert classify_gray(crop(img, 64, 64, 62))[0] == kind


@pytest.mark.parametrize('kind', ['no_parking', 'no_stopping'])
def test_classify_tolerates_radius_error(kind):
    img = render_sign(kind, 128)
    for r in (55, 62, 68):   # Hough rarely gets the radius exactly
        assert classify_color(crop(img, 64, 64, r))[0] == kind, r


def test_mirrored_stripe_is_not_a_no_parking_sign():
    img = cv2.flip(render_sign('no_parking', 128), 1)   # stripe / instead of \
    assert classify_color(crop(img, 64, 64, 62))[0] is None


def test_plain_red_disc_is_rejected():
    img = np.full((128, 128, 3), 200, np.uint8)
    cv2.circle(img, (64, 64), 62, (28, 18, 193), -1)
    assert classify_color(crop(img, 64, 64, 62))[0] is None


@pytest.mark.parametrize('kind', ['no_parking', 'no_stopping'])
@pytest.mark.parametrize('gray', [False, True])
def test_detector_finds_sign_in_scene(kind, gray):
    [d] = SignDetector()(scene(kind, gray=gray, noise=4))
    assert d.label == kind
    cx, cy, r = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2, (d.box[2] - d.box[0]) / 2
    assert abs(cx - 400) < 4 and abs(cy - 150) < 4 and abs(r - 30) < 5


def test_small_far_sign_is_found():
    [d] = SignDetector()(scene('no_parking', r=10, noise=3))
    assert d.label == 'no_parking'


def test_empty_scene_has_no_signs():
    assert SignDetector()(scene(None, noise=4)) == []


def test_distance_from_size():
    # 0.2 m sign, fx 500: at 2 m it is 500 * 0.2 / 2 = 50 px wide -> radius 25 px.
    p = pixel_to_point_by_size(320, 180, 25, 0.2, INTR, np.eye(3), np.zeros(3))
    assert np.allclose(p, [0, 0, 2.0])
    assert pixel_to_point_by_size(320, 180, 1, 0.2, INTR, np.eye(3), np.zeros(3), max_range=10) is None
