import math

import numpy as np

from parking_rules.geometry import pixel_ray, pixel_to_ground, quat_to_matrix, scaled_intrinsics

# Optical frame (x right, y down, z forward) mounted on a link with x forward, y left, z up:
# optical z = link x, optical x = -link y, optical y = -link z.
OPTICAL_TO_LINK = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
INTR = (500.0, 500.0, 320.0, 180.0)


def test_quat_identity_and_yaw():
    assert np.allclose(quat_to_matrix(0, 0, 0, 1), np.eye(3))
    R = quat_to_matrix(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # yaw 90°
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_principal_point_is_straight_ahead():
    assert np.allclose(pixel_ray(320, 180, *INTR), [0, 0, 1])


def test_scaled_intrinsics_half_resolution():
    k = [500, 0, 320, 0, 500, 180, 0, 0, 1]
    assert scaled_intrinsics(k, (640, 360), (320, 180)) == (250, 250, 160, 90)


def test_ground_point_ahead_of_camera():
    # Camera 0.3 m above the ground looking straight ahead. A pixel 50 px below the centre looks down by
    # atan(50/500), so it hits the ground at 0.3 / (50/500) = 3 m.
    cam_t = np.array([0.0, 0.0, 0.3])
    pt = pixel_to_ground(320, 230, INTR, OPTICAL_TO_LINK, cam_t, np.eye(3), np.zeros(3))
    assert np.allclose(pt, [3.0, 0.0, 0.0])


def test_pixel_to_the_right_lands_right():
    cam_t = np.array([0.0, 0.0, 0.3])
    pt = pixel_to_ground(420, 230, INTR, OPTICAL_TO_LINK, cam_t, np.eye(3), np.zeros(3))
    assert pt[1] < 0  # y left, so right is negative
    assert np.isclose(pt[0], 3.0)


def test_above_horizon_never_hits_ground():
    cam_t = np.array([0.0, 0.0, 0.3])
    assert pixel_to_ground(320, 100, INTR, OPTICAL_TO_LINK, cam_t, np.eye(3), np.zeros(3)) is None


def test_too_far_is_rejected():
    cam_t = np.array([0.0, 0.0, 0.3])
    assert pixel_to_ground(320, 181, INTR, OPTICAL_TO_LINK, cam_t, np.eye(3), np.zeros(3), max_range=10) is None
