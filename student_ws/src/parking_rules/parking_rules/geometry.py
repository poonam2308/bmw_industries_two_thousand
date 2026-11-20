"""Pixel -> point on the ground. Pure numpy, no ROS, so it can be unit-tested.

Idea: an object standing on the ground (a hydrant) touches it at the bottom centre of its bounding box. Shoot a ray
from the camera through that pixel and intersect it with the ground plane. This needs no depth image, which matters
because the depth topic is far too big to stream over the car's Wi-Fi. It assumes flat ground and gets less precise
far away (one pixel covers more metres), so the tracker averages over many frames.
"""
import numpy as np


def quat_to_matrix(x, y, z, w):
    """3x3 rotation matrix from a unit quaternion."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_to_rt(t):
    """geometry_msgs/Transform (or anything with .translation / .rotation) -> (R, t)."""
    q, p = t.rotation, t.translation
    return quat_to_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])


def scaled_intrinsics(k, info_size, image_size):
    """fx, fy, cx, cy from a CameraInfo K (row-major 3x3), rescaled when the image (e.g. the low-rate preview) has
    a different resolution than the camera_info."""
    (iw, ih), (w, h) = info_size, image_size
    sx, sy = w / iw, h / ih
    return k[0] * sx, k[4] * sy, k[2] * sx, k[5] * sy


def pixel_ray(u, v, fx, fy, cx, cy):
    """Direction of the ray through pixel (u, v) in the optical frame (x right, y down, z forward)."""
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return d / np.linalg.norm(d)


def ray_plane(origin, direction, plane_point, plane_normal, max_range=15.0):
    """Intersection of a ray with a plane, or None if the ray points away from it or hits it too far away."""
    denom = float(np.dot(plane_normal, direction))
    if abs(denom) < 1e-6:
        return None  # parallel to the ground (looking at the horizon)
    s = float(np.dot(plane_normal, plane_point - origin)) / denom
    if s <= 0 or s > max_range:
        return None
    return origin + s * direction


def pixel_to_point_by_size(u, v, radius_px, diameter_m, intrinsics, cam_R, cam_t, max_range=15.0):
    """Centre of a round object of known size (a sign) -> 3D point in the fixed frame, or None if too far.

    Pinhole camera: an object of diameter D at depth z appears 2 * radius_px = fx * D / z pixels wide.
    Works for objects that do not touch the ground (a sign on a pole). Precision depends on the circle radius: at
    10 px a 1 px error is 10 % of the distance, so the tracker averages.
    """
    fx, fy, cx, cy = intrinsics
    if radius_px <= 0:
        return None
    z = fx * diameter_m / (2 * radius_px)
    p = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
    if np.linalg.norm(p) > max_range:
        return None
    return cam_R @ p + cam_t


def yaw_of_matrix(R):
    """Heading (rotation about z) of a frame from its rotation matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def pixel_to_ground(u, v, intrinsics, cam_R, cam_t, ground_R, ground_t, max_range=15.0):
    """Pixel -> 3D point on the ground, all expressed in one fixed frame (e.g. `odom`).

    cam_R, cam_t: pose of the camera optical frame in the fixed frame.
    ground_R, ground_t: pose of a frame whose xy-plane is the ground (`base_footprint`) in the fixed frame.
    """
    ray = cam_R @ pixel_ray(u, v, *intrinsics)
    return ray_plane(cam_t, ray, ground_t, ground_R[:, 2], max_range)
