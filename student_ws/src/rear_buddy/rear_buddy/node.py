#!/usr/bin/env python3
"""Rear Buddy on the car (or a bag replay): rear camera in, reactions out.

    ros2 run rear_buddy rear_buddy_node
    ros2 run rear_buddy rear_buddy_node --ros-args -p fancy:=always -p rate_hz:=2.0
    ros2 run rear_buddy rear_buddy_node --ros-args -p flirt:=true     # demo only

Subscribes  /back_camera/image/compressed   (parameter: topic)
            /wheel_odom                     how long we have been standing still (for the 'stuck' texts)
            /batt_status                    battery voltage (for texts that mention it)
Publishes   /rear_buddy/reaction            std_msgs/String, JSON {scene, text, animation}, whenever it changes
            (the rear LEDs and the phone display will listen to this later)
"""
import json
import time

import cv2
import numpy as np
import rclpy
from car_interfaces.msg import BattStatus, WheelOdom
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .detector import Detector
from .fancy import make_fancy_check
from .pipeline import analyse
from . import rules
from .rules import Director

STOPPED_MPS = 0.05  # below this wheel speed the car counts as standing still
STALE_S = 3.0       # wheel or battery data older than this is ignored


class RearBuddy(Node):
    def __init__(self):
        super().__init__('rear_buddy')
        topic = self.declare_parameter('topic', '/back_camera/image/compressed').value
        model = self.declare_parameter('model', 'yolo11n.pt').value
        fancy = self.declare_parameter('fancy', 'clip').value  # car brand check: clip, always, budget, never
        rate_hz = self.declare_parameter('rate_hz', 2.0).value
        flirt = self.declare_parameter('flirt', False).value  # demo only: the car flirts with cyclists
        rules.PITCH_PEOPLE = self.declare_parameter('pitch_people', False).value  # people get the job pitch

        self.detector = Detector(model)
        self.fancy = make_fancy_check(fancy, log=self.get_logger().info)
        self.director = Director(flirt=flirt)
        self.latest_image = None
        self.stopped_since = None
        self.wheels_at = self.battery_at = 0.0
        self.battery_v = None

        best_effort = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage, topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(WheelOdom, '/wheel_odom', self.on_wheels, best_effort)
        self.create_subscription(BattStatus, '/batt_status', self.on_battery, best_effort)
        self.pub = self.create_publisher(String, '/rear_buddy/reaction', 10)
        # Detection runs on a timer on the newest frame only, so a slow laptop skips frames instead of lagging.
        self.create_timer(1.0 / rate_hz, self.step)
        self.get_logger().info(f'waiting for {topic} ...')

    def on_image(self, msg):
        self.latest_image = msg

    def on_wheels(self, msg):
        now = time.monotonic()
        self.wheels_at = now
        if max(abs(msg.left_speed_mps), abs(msg.right_speed_mps)) < STOPPED_MPS:
            self.stopped_since = self.stopped_since or now
        else:
            self.stopped_since = None

    def on_battery(self, msg):
        self.battery_v, self.battery_at = msg.voltage_total, time.monotonic()

    def step(self):
        msg, self.latest_image = self.latest_image, None
        if msg is None:
            return
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('could not decode image')
            return
        now = time.monotonic()
        stopped_s = None
        if now - self.wheels_at < STALE_S:
            stopped_s = now - self.stopped_since if self.stopped_since else 0.0
        battery_v = self.battery_v if now - self.battery_at < STALE_S * 2 else None  # 1 Hz topic

        scene, _, _ = analyse(frame, self.detector, self.fancy)
        reaction = self.director.update(scene, stopped_s, battery_v)
        if reaction:
            self.get_logger().info(f'[{reaction.scene}] leds={reaction.animation} {reaction.text!r}')
            self.pub.publish(String(data=json.dumps(reaction.__dict__)))


def main():
    rclpy.init()
    node = RearBuddy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
