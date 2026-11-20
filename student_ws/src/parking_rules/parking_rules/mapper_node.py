"""ROS node: detect things that make parking illegal, place them on the map and publish no-parking zones.

    ros2 run parking_rules parking_rules_mapper
    ros2 run parking_rules parking_rules_mapper --ros-args -p image_topic:=/visual_slam/image_0   # simulation
    ros2 run parking_rules parking_rules_mapper --ros-args -p detect_hydrants:=false              # signs only
    ros2 run parking_rules parking_rules_mapper --ros-args -p plate_reader:=claude                # needs internet

camera image ─┬▶ YOLO "fire hydrant" ─▶ bottom-centre pixel ─▶ ray ∩ ground ─────────────┐
              ├▶ signs.py round signs (286/283) ─┐                                          │
              └▶ signs.py P sign (314) ──────────┴▶ centre + size ─▶ distance by size ──────┤ (tf, image time)
                     │                                                                     ▼
                     └▶ find_plate ─▶ reader thread (tesseract / claude) ─▶ PlateBook (vote)
                        Tracker (odom frame; for signs also car heading + left/right side) ─▶ zones.py
                        ─▶ /parking_rules/zones (JSON), /parking_rules/markers (RViz / Foxglove)

Publishes
    /parking_rules/zones            std_msgs/String (JSON list of zones, transient local: late subscribers get it)
    /parking_rules/markers          visualization_msgs/MarkerArray (red = forbidden, amber = parking with conditions,
                                    green = parking area; labels show the plate's schedule)
    /parking_rules/debug/compressed sensor_msgs/CompressedImage (camera frame with boxes)
"""
from concurrent.futures import ThreadPoolExecutor
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .detector import Detection, Detector, draw
from .geometry import pixel_to_ground, pixel_to_point_by_size, scaled_intrinsics, transform_to_rt, yaw_of_matrix
from .plate_book import PlateBook
from .plates import Schedule
from .readers import make_reader
from .signs import (LABELS as SIGN_LABELS, PARKING, ParkingSignDetector, SignDetector, detect_all_signs,
                    find_plate, is_color)
from .tracker import Tracker
from .zones import RULES, StripRule, zones_from_landmarks, zones_to_json


def decode(msg):
    """CompressedImage or Image -> HxWx3 BGR array (YOLO wants 3 channels; the stereo cameras are mono)."""
    if isinstance(msg, CompressedImage):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    channels = {'mono8': 1, 'rgb8': 3, 'bgr8': 3, 'bgra8': 4, 'rgba8': 4}.get(msg.encoding)
    if channels is None:
        return None
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * channels]
    a = a.reshape(msg.height, msg.width, channels)
    conv = {'mono8': cv2.COLOR_GRAY2BGR, 'rgb8': cv2.COLOR_RGB2BGR, 'bgra8': cv2.COLOR_BGRA2BGR,
            'rgba8': cv2.COLOR_RGBA2BGR}.get(msg.encoding)
    return cv2.cvtColor(a, conv) if conv is not None else a.copy()


class ParkingRulesMapper(Node):
    def __init__(self):
        super().__init__('parking_rules_mapper')
        p = self.declare_parameter
        self.image_topic = p('image_topic', '/preview/left/compressed').value
        info_topic = p('camera_info_topic', '/visual_slam/camera_info_0').value
        self.camera_frame = p('camera_frame', 'left_optical').value  # used when the image header has none
        self.fixed_frame = p('fixed_frame', 'odom').value             # Nav2's global frame on BITT
        self.ground_frame = p('ground_frame', 'base_footprint').value  # its xy-plane is the ground
        self.rate_hz = p('rate_hz', 3.0).value                         # detectors run at most this often
        self.max_range = p('max_range', 8.0).value                     # ignore detections further away (m)
        self.sign_diameter = p('sign_diameter', 0.20).value            # m, outer diameter of YOUR printed sign
        self.parking_sign_size = p('parking_sign_size', 0.20).value    # m, side of YOUR printed P sign
        detect_hydrants = p('detect_hydrants', True).value
        detect_signs = p('detect_signs', True).value
        detect_parking_signs = p('detect_parking_signs', True).value
        reader_kind = p('plate_reader', 'tesseract').value             # tesseract | claude | fixed | none
        plate_text = p('plate_text', '').value                         # for plate_reader:=fixed (simulation)
        max_plate_reads = p('max_plate_reads', 5).value
        zone_length = p('sign_zone_length', RULES['no_parking'].length).value  # m along the road from the sign
        zone_width = p('sign_zone_width', RULES['no_parking'].width).value     # m from the sign into the road
        self.rules = dict(RULES)
        for label in SIGN_LABELS + (PARKING,):
            self.rules[label] = StripRule(zone_length, zone_width, RULES[label].reason)

        self.hydrants = Detector(p('model', 'yolo11n.pt').value, p('min_conf', 0.35).value) if detect_hydrants else None
        self.signs = SignDetector(min_radius=p('sign_min_radius_px', 6).value) if detect_signs else None
        self.parking = ParkingSignDetector() if detect_parking_signs else None
        try:
            self.reader = make_reader(reader_kind, log=self.get_logger().warning, fixed_text=plate_text)
        except (RuntimeError, ValueError) as e:
            self.reader = None
            self.get_logger().error(f'{e}\n -> extra plates cannot be read; signs with a plate count as illegal')
        self.plates = PlateBook(max_reads=max_plate_reads)
        self.pool = ThreadPoolExecutor(max_workers=1)   # plate reading is slow (OCR / API): off the image callback
        self.dirty = False
        self.create_timer(1.0, self.on_timer)
        self.tracker = Tracker(gate=p('gate', 0.75).value, min_hits=p('min_hits', 3).value)
        self.info = None
        self.last_run = 0.0
        self.warned_gray = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        msg_type = CompressedImage if self.image_topic.endswith('compressed') else Image
        self.create_subscription(msg_type, self.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.zones_pub = self.create_publisher(String, '/parking_rules/zones', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/parking_rules/markers', latched)
        self.debug_pub = self.create_publisher(CompressedImage, '/parking_rules/debug/compressed', 1)
        self.get_logger().info(f'waiting for {self.image_topic} and {info_topic} ...')

    def on_info(self, msg):
        if self.info is None:
            self.get_logger().info(f'camera_info {msg.width}x{msg.height} received')
        self.info = msg

    def on_image(self, msg):
        now = time.monotonic()
        if self.info is None or now - self.last_run < 1.0 / self.rate_hz:
            return
        self.last_run = now
        frame = decode(msg)
        if frame is None:
            self.get_logger().warning(f'cannot decode image ({getattr(msg, "encoding", "jpeg")})')
            return

        hydrants = self.hydrants(frame) if self.hydrants else []
        signs = detect_all_signs(frame, self.signs, self.parking)
        plate_boxes = [find_plate(frame, d.box) for d in signs]
        if self.signs and not self.warned_gray and not is_color(frame):
            self.warned_gray = True
            self.get_logger().warning('grayscale camera image: round signs (286/283) are told apart only weakly '
                                      '(red and blue look alike in gray), check them in the debug image; '
                                      'P signs and plates are fine in gray')
        self.publish_debug(msg.header, frame,
                           hydrants + signs + [Detection('plate', 1.0, b) for b in plate_boxes if b])
        if not hydrants and not signs:
            return

        cam_frame = msg.header.frame_id or self.camera_frame
        stamp = Time.from_msg(msg.header.stamp)
        try:
            # Both poses at the time the image was taken, so driving while detecting does not smear the result.
            cam = self.tf_buffer.lookup_transform(self.fixed_frame, cam_frame, stamp, Duration(seconds=0.2))
            ground = self.tf_buffer.lookup_transform(self.fixed_frame, self.ground_frame, stamp,
                                                     Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().warning(f'no transform: {e}', throttle_duration_sec=5.0)
            return

        h, w = frame.shape[:2]
        intr = scaled_intrinsics(self.info.k, (self.info.width, self.info.height), (w, h))
        cam_R, cam_t = transform_to_rt(cam.transform)
        g_R, g_t = transform_to_rt(ground.transform)
        car_yaw = yaw_of_matrix(g_R)
        t = stamp.nanoseconds * 1e-9

        sightings = []
        for d in hydrants:
            pt = pixel_to_ground(*d.foot, intr, cam_R, cam_t, g_R, g_t, self.max_range)
            if pt is not None:
                sightings.append((d.label, pt, None, None, None))
        for d, plate_box in zip(signs, plate_boxes):
            size = self.parking_sign_size if d.label == PARKING else self.sign_diameter
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            pt = pixel_to_point_by_size(cx, cy, (d.box[2] - d.box[0]) / 2, size, intr,
                                        cam_R, cam_t, self.max_range)
            if pt is not None:
                # Left (+y in the car frame) or right of the car decides on which side of the road the rule applies.
                lateral = -math.sin(car_yaw) * (pt[0] - g_t[0]) + math.cos(car_yaw) * (pt[1] - g_t[1])
                sightings.append((d.label, pt, car_yaw, 1 if lateral > 0 else -1, (d, plate_box)))

        for label, pt, heading, side, sign in sightings:
            lm = self.tracker.update(label, float(pt[0]), float(pt[1]), t, heading, side)
            if lm.hits == self.tracker.min_hits:
                where = '' if lm.side == 0 else (' (left of the road)' if lm.side > 0 else ' (right of the road)')
                self.get_logger().info(f'{lm.label} #{lm.id} confirmed at ({lm.x:.2f}, {lm.y:.2f}) '
                                       f'in {self.fixed_frame}{where}')
            if sign and sign[1]:
                self.plates.saw_plate(lm.id)
                # Read only confirmed signs (no OCR / API calls for one-off false detections).
                if self.reader and lm.hits >= self.tracker.min_hits and self.plates.start_read(lm.id):
                    self.pool.submit(self.read_plate, lm.id, *self.crops(frame, *sign))
        if sightings:
            self.publish_zones()

    @staticmethod
    def crops(frame, sign, plate_box):
        """(sign with its plates, plates only) as image copies for the reader thread."""
        H, W = frame.shape[:2]
        x1, y1 = min(sign.box[0], plate_box[0]), sign.box[1]
        x2, y2 = max(sign.box[2], plate_box[2]), plate_box[3]
        m = 0.05 * (x2 - x1)
        clip = lambda v, hi: int(min(max(v, 0), hi))                       # noqa: E731
        both = frame[clip(y1 - m, H):clip(y2 + m, H), clip(x1 - m, W):clip(x2 + m, W)].copy()
        plate = frame[clip(plate_box[1], H):clip(plate_box[3], H), clip(plate_box[0], W):clip(plate_box[2], W)].copy()
        return both, plate

    def read_plate(self, sign_id, both, plate):   # runs in the reader thread
        try:
            text = self.reader(both, plate)
        except Exception as e:                     # a reader bug must not kill the thread pool silently
            self.get_logger().error(f'plate reader failed: {e!r}')
            text = None
        schedule = self.plates.finish_read(sign_id, text)
        shown = (text or '').replace('\n', ' | ')
        self.get_logger().info(f'plate of sign #{sign_id}: "{shown}" -> '
                               f'{schedule.describe() if schedule else "not understood"}')
        self.dirty = True

    def on_timer(self):
        if self.dirty:
            self.dirty = False
            self.publish_zones()

    def publish_zones(self):
        confirmed = self.tracker.confirmed()
        zones = zones_from_landmarks(confirmed, self.rules, self.plates.all_status([lm.id for lm in confirmed]))
        self.zones_pub.publish(String(data=zones_to_json(zones)))
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, z in enumerate(zones):
            area = Marker(ns='no_parking', id=i, action=Marker.ADD)
            area.header.frame_id = self.fixed_frame
            area.pose.orientation.w = 1.0
            if z.kind == PARKING and z.plate == 'none':
                area.color.g, area.color.b, area.color.a = 0.7, 0.3, 0.4   # parking allowed: green
            elif z.kind == PARKING:
                area.color.r, area.color.g, area.color.a = 1.0, 0.7, 0.4   # allowed with conditions: amber
            else:
                area.color.r, area.color.a = 1.0, 0.4                      # forbidden: red
            if z.polygon:
                area.type = Marker.TRIANGLE_LIST
                area.scale.x = area.scale.y = area.scale.z = 1.0
                pts = [Point(x=float(x), y=float(y), z=0.02) for x, y in z.polygon]
                for k in range(1, len(pts) - 1):  # fan triangulation (the strip is convex)
                    area.points += [pts[0], pts[k], pts[k + 1]]
            else:
                area.type = Marker.CYLINDER
                area.pose.position = Point(x=z.x, y=z.y, z=0.02)
                area.scale.x = area.scale.y = 2 * z.radius
                area.scale.z = 0.04
            label = Marker(ns='no_parking_label', id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            label.header.frame_id = self.fixed_frame
            label.pose.position = Point(x=z.x, y=z.y, z=0.6)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.2
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = z.reason
            if z.schedule:
                label.text += f' [{Schedule.from_dict(z.schedule).describe()}]'
            elif z.plate == 'unread':
                label.text += ' [plate not read yet]'
            markers.markers += [area, label]
        self.marker_pub.publish(markers)

    def publish_debug(self, header, frame, detections):
        if self.debug_pub.get_subscription_count() == 0:
            return
        ok, jpg = cv2.imencode('.jpg', draw(frame, detections), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.debug_pub.publish(CompressedImage(header=header, format='jpeg', data=jpg.tobytes()))


def main():
    rclpy.init()
    node = ParkingRulesMapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
