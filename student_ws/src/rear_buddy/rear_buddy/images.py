"""Camera messages -> OpenCV images (BGR numpy arrays), without ROS.

The real car sends JPEG (sensor_msgs/CompressedImage), the simulation raw pixels (sensor_msgs/Image).
"""
import cv2
import numpy as np

from .cdr import normalize

COMPRESSED = 'sensor_msgs/CompressedImage'
RAW = 'sensor_msgs/Image'
CONVERSIONS = {  # encoding: (channels, OpenCV conversion to BGR or None)
    'bgr8': (3, None), 'rgb8': (3, cv2.COLOR_RGB2BGR), 'bgra8': (4, cv2.COLOR_BGRA2BGR),
    'rgba8': (4, cv2.COLOR_RGBA2BGR), 'mono8': (1, cv2.COLOR_GRAY2BGR),
}


def is_image_type(msg_type):
    return normalize(msg_type or '') in (COMPRESSED, RAW)


def to_bgr(codec, msg_type, payload):
    """Decode one camera message. Returns (image or None, problem description or None)."""
    msg_type = normalize(msg_type or COMPRESSED)
    msg = codec.decode(msg_type, payload)
    if msg_type == COMPRESSED:
        frame = cv2.imdecode(np.frombuffer(msg['data'], np.uint8), cv2.IMREAD_COLOR)
        return (frame, None) if frame is not None else (None, f'cannot decode a {msg["format"]!r} image')
    encoding = msg['encoding'].lower()
    if encoding not in CONVERSIONS:
        return None, f'unsupported image encoding {encoding!r}'
    channels, conversion = CONVERSIONS[encoding]
    h, w, step = msg['height'], msg['width'], msg['step']
    rows = np.frombuffer(msg['data'], np.uint8)[:h * step].reshape(h, step)
    image = rows[:, :w * channels].reshape(h, w, channels)
    return (cv2.cvtColor(image, conversion) if conversion is not None else image.copy()), None
